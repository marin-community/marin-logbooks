# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fresh plain-v0.1 replay capture using the unchanged proven T2 raw-token primitive."""

import argparse
import hashlib
import json
import os
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from experiments.post_training.identity_path_b.checkpoints import TEMPORARY_PREFIX
from experiments.post_training.identity_t2 import capture as t2

T2_CAPTURE_SHA256 = "3ef731cfe8075677a41849174d850fefeea9ab7bc821d30cfd9ecacd304b6f67"
SOURCE_MODEL = "s3://marin-rno2a/marin/models/marin-snowball-67b-a2b-v0.1-adc0078a"
SOURCE_REVISION = "adc0078a19a4383bc998fcdb43867b3360baf5f3"


def checked_rows(manifest: Path, expected_sha256: str) -> list[dict]:
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Fresh replay manifest hash changed")
    if hashlib.sha256(Path(t2.__file__).read_bytes()).hexdigest() != T2_CAPTURE_SHA256:
        raise ValueError("Known-good T2 capture primitive changed")
    rows = [json.loads(line) for line in raw.splitlines()]
    if (
        len(rows) != 12000
        or len({row["id"] for row in rows}) != 12000
        or len({row["source_id"] for row in rows}) != 6000
    ):
        raise ValueError("Expected exactly6000 fresh originals/12000 paired-mode replay requests")
    variants = {}
    for row in rows:
        if (
            row["teacher"] != "plain"
            or row["teacher_system"]
            or any(message["role"] == "system" for message in row["prompt"])
        ):
            raise ValueError("Replay must be plainv0.1 without identity teacher/system conditioning")
        pair = variants.setdefault(row["source_id"], {})
        if row["enable_thinking"] in pair:
            raise ValueError("Repeated replay source/mode")
        pair[row["enable_thinking"]] = row
    for pair in variants.values():
        if (
            set(pair) != {False, True}
            or pair[False]["prompt"] != pair[True]["prompt"]
            or pair[False]["tools"] != pair[True]["tools"]
        ):
            raise ValueError("Replay source off/on pair changed context or tools")
    if len({row["source_id"] for row in rows[:12]}) != 12:
        raise ValueError("Live12 must use12 distinct original sources")
    if {(row["shape"], row["enable_thinking"]) for row in rows[:12]} != {
        (shape, mode)
        for shape in ("math", "code", "chat", "call_after_result", "first_call", "reply")
        for mode in (False, True)
    }:
        raise ValueError("Live12 must cover both modes of all3 general families and all3 tool shapes")
    return rows


def serving_row(row: dict) -> dict:
    """Match native serving's typed tool-wrapper order before CPU rendering.

    The function parameters remain the original untyped JSON mapping, including
    its key order. No schema value, prompt message, generated text or raw ID is
    edited. The original immutable manifest remains the selection authority.
    """
    tools = []
    for tool in row["tools"]:
        function = tool["function"]
        if set(tool) != {"type", "function"} or set(function) != {"name", "description", "parameters"}:
            raise ValueError("Unexpected tool fields outside the proven native serving schema")
        tools.append(
            {"type": tool["type"], "function": {key: function[key] for key in ("name", "description", "parameters")}}
        )
    return {**row, "tools": tools}


def native_preflight(rows: list[dict], tokenizer) -> dict:
    sizes = []
    for original in rows:
        row = serving_row(original)
        prefix = t2.prompt_ids(tokenizer, row)
        if prefix != t2.prompt_ids(tokenizer, row, teacher=True) or len(prefix) >= 8192:
            raise ValueError("Plain teacher/student native prefix mismatch or impossible8192-token training row")
        sizes.append(len(prefix))
    return {
        "rows": len(rows),
        "minimum_native_prefix_tokens": min(sizes),
        "maximum_native_prefix_tokens": max(sizes),
        "plain_teacher_student_prefix_id_equality": True,
    }


def pilot_passes(records: list[dict]) -> bool:
    """Protocol/tool proof only: no style-D teacher, identity question, or name-first gate."""
    if len(records) != 12 or any(row.get("fatal") or row.get("http_status") != 200 for row in records):
        return False
    modes_with_valid_call = set()
    for record in records:
        row = record["candidate"]
        if (
            not record.get("roundtrip", {}).get("ok")
            or record.get("length_stop")
            or record.get("fidelity_rejected")
            or row["family"] != "tool"
            or t2.TOOL_OPEN_ID not in record.get("raw_token_ids", [])
        ):
            continue
        final = record["roundtrip"]["content"].rsplit("<|end_think|>", 1)[-1]
        names = {tool["function"]["name"] for tool in row["tools"]}
        for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", final, re.S):
            try:
                call = json.loads(match.group(1))
            except ValueError:
                continue
            if isinstance(call, dict) and call.get("name") in names and isinstance(call.get("arguments"), dict):
                modes_with_valid_call.add(row["enable_thinking"])
    return modes_with_valid_call == {False, True}


def capture(rows: list[dict], *, tokenizer, endpoint: str, model: str, output: Path, artifact_root: str) -> dict:
    if not artifact_root.startswith(TEMPORARY_PREFIX):
        raise ValueError("Replay captures require the RNO14-day TTL artifact root")
    if not endpoint.startswith("https://") or not endpoint.endswith("/v1"):
        raise ValueError("Require the owned TLS-protected native T2 endpoint ending/v1")
    preflight = native_preflight(rows, tokenizer)
    output.mkdir(parents=True, exist_ok=False)
    rows = [serving_row(row) for row in rows]
    records = []
    budget = t2.FidelityBudget()
    stop = False

    def observe(record: dict) -> bool:
        records.append(record)
        # A tokenizer mismatch occurs BEFORE a completion attempt, so it is
        # immediately fatal but never inflates the output-rate denominator.
        if record.get("server_teacher_prompt_token_ids") != record["cpu_teacher_prompt_token_ids"]:
            budget.fatal_protocol = True
            return True
        return budget.observe(record)

    for index, row in enumerate(rows[:12]):
        record = t2.capture_one(index, row, endpoint=endpoint, model=model, tokenizer=tokenizer, out=output)
        if observe(record):
            stop = True
            break
    first = list(records)
    pilot = {
        "passes": pilot_passes(first),
        "output_fidelity_rate_policy": "100 attempts, every100, and terminal; NOT at live12",
        "output_fidelity_denominator": "every attempted completion record incl drops, length cuts and HTTP failures",
        "output_fidelity_numerator": "output-fidelity rejects only",
        "prompt_id_mismatch": "fatal immediately; no completion requested",
        "records": [
            {
                "id": row["candidate"]["id"],
                "fatal": row.get("fatal"),
                "length_stop": row.get("length_stop"),
                "fidelity_rejected": row.get("fidelity_rejected", False),
                "roundtrip": row.get("roundtrip", {}).get("ok"),
            }
            for row in first
        ],
    }
    (output / "live12-receipt.json").write_text(json.dumps(pilot, indent=2) + "\n")
    if pilot["passes"]:
        # At most8 HTTP attempts are in flight. Once a fatal or >5% rate is
        # observed, issue no new requests; retain and drain existing attempts.
        with ThreadPoolExecutor(max_workers=8) as executor:
            pending = {}
            next_index = 12
            while (next_index < len(rows) and not stop) or pending:
                while not stop and next_index < len(rows) and len(pending) < 8:
                    future = executor.submit(
                        t2.capture_one,
                        next_index,
                        rows[next_index],
                        endpoint=endpoint,
                        model=model,
                        tokenizer=tokenizer,
                        out=output,
                    )
                    pending[future] = next_index
                    next_index += 1
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda future: pending[future]):
                    pending.pop(future)
                    stop = observe(future.result()) or stop
    over_budget = budget.finish()
    records.sort(key=lambda row: row["index"])
    report = {
        "status": "CAPTURE_ONLY_NOT_ACCEPTED_REPLAY_OR_TRAINING_CLEAR",
        "source_model": SOURCE_MODEL,
        "source_revision": SOURCE_REVISION,
        "api_model_name": model,
        "t2_capture_sha256": T2_CAPTURE_SHA256,
        "artifact_root": artifact_root,
        "preflight": preflight,
        "attempts": budget.attempted,
        "capture_records": len(records),
        "live12_passes": pilot["passes"],
        "output_fidelity_rate_checkpoints": budget.checkpoints,
        "output_fidelity_over_5_percent": over_budget,
        "fatal_protocol_or_prompt": budget.fatal_protocol,
        "requests_not_started": len(rows) - len(records),
        "inflight_policy": "max8; stop new requests immediately on observed fatal/budget breach; drain all issued",
        "attempts_per_request": 1,
        "retries": 0,
        "fatal": sum(bool(row.get("fatal")) for row in records),
        "length_stop": sum(bool(row.get("length_stop")) for row in records),
        "fidelity_rejected": sum(bool(row.get("fidelity_rejected")) for row in records),
        "raw_capture_sha256": {
            f"{row['index']:04d}.json": hashlib.sha256((output / f"{row['index']:04d}.json").read_bytes()).hexdigest()
            for row in records
        },
        "note": "Raw IDs/EOT stay authoritative; row QC, accepted counts and actual packed doses follow separately.",
    }
    (output / "capture-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    if not pilot["passes"] or budget.fatal_protocol or over_budget:
        raise RuntimeError("Capture stopped at the fixed prompt/protocol/tool gate or output-fidelity budget; no retry")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prove", "capture"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--artifact-root")
    args = parser.parse_args()
    rows = checked_rows(args.manifest, args.manifest_sha256)
    tokenizer = t2.checked_tokenizer(args.tokenizer_dir)
    if args.action == "prove":
        report = native_preflight(rows, tokenizer)
        report["manifest_sha256"] = args.manifest_sha256
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    else:
        if not args.model or not args.artifact_root:
            parser.error("capture requires --model and --artifact-root; endpoint comes from PATH_B_REPLAY_ENDPOINT env")
        report = capture(
            rows,
            tokenizer=tokenizer,
            endpoint=os.environ["PATH_B_REPLAY_ENDPOINT"],
            model=args.model,
            output=args.output,
            artifact_root=args.artifact_root,
        )
    summary = (
        {key: report[key] for key in ("rows", "minimum_native_prefix_tokens", "maximum_native_prefix_tokens")}
        if args.action == "prove"
        else {key: report[key] for key in ("status", "attempts", "fatal", "length_stop", "fidelity_rejected")}
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
