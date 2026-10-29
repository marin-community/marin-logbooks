# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-attempt, 22-source GLM visible-rationale pilot; never approves a training pool."""

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from marin.inference.openai_chat import OpenAIChatClient

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL, resolve_glm_base_url
from experiments.post_training.identity_path_b.checkpoints import TEMPORARY_PREFIX
from experiments.post_training.identity_path_b.data import SPECIAL_TOKEN_IDS
from experiments.post_training.identity_path_b.reasoning import graft_reasoning, identity_qc
from experiments.post_training.identity_t2.capture import checked_tokenizer

RELAY_JOB = "/muchanem/glm53-relay-rno2a-d"
CAPTURE_CLUSTER = "cw-rno2a"
CANONICAL_SHA256 = "d430314e7bc9b950a5a0fb0e8dc492f74c726ef75c984de8e3ba4b35eae6ab71"
CONTEXT_SHA256 = "01e148c75cf658fb5b5c2bfa28f0df6a0ae58465a5dbe151a4f12e3e6f28a729"
TRACE_FIELD = "choices[0].message.content"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_sources(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if sha(raw) != CANONICAL_SHA256:
        raise ValueError("GLM pilot requires the fixed 22 canonical original sources")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 22 or len({row["source_id"] for row in rows}) != 22 or any(row["enable_thinking"] for row in rows):
        raise ValueError("Expected exactly22 distinct canonical OFF sources")
    return rows


def pilot_requests(canonical: Path, context_file: Path) -> list[dict]:
    context = context_file.read_text().removesuffix("\n")
    if sha(context.encode()) != CONTEXT_SHA256:
        raise ValueError("GLM pilot context differs from the review packet")
    requests = []
    for row in canonical_sources(canonical):
        source = row["source_id"]
        payload = {"conversation": row["messages"][:-1], "canonical_final_answer": row["messages"][-1]["content"]}
        requests.append(
            {
                "id": f"pathb-glm-{source}",
                "source_id": source,
                "family": row["family"],
                "canonical_off_id": row["id"],
                "canonical_off_sha256": sha(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()),
                "request_body": {
                    "model": GLM_MODEL,
                    "messages": [
                        {"role": "system", "content": context},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
                    ],
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_tokens": 1024,
                    "seed": int(sha(f"310:glm-reason:{source}".encode())[:8], 16) % (2**31),
                    "n": 1,
                    "stream": False,
                    "tool_choice": "none",
                    "chat_template_kwargs": {"reasoning_effort": "high"},
                },
            }
        )
    return requests


def checked_manifest(path: Path, expected_sha256: str, canonical: Path, context: Path) -> list[dict]:
    raw = path.read_bytes()
    if sha(raw) != expected_sha256:
        raise ValueError("GLM pilot manifest hash changed")
    rows = [json.loads(line) for line in raw.splitlines()]
    if rows != pilot_requests(canonical, context):
        raise ValueError("GLM manifest differs from the bound sources/context/settings")
    return rows


def capture_one(client: OpenAIChatClient, request: dict, index: int, output: Path, base_url: str) -> dict:
    record = {"request": request, "relay_job": RELAY_JOB, "base_url": base_url, "trace_field": TRACE_FIELD}
    try:
        record["response"] = client.complete(request["request_body"])
    except Exception as error:
        # Preserve each terminal failed attempt in the acceptance denominator;
        # never retry it or serialize exception text that could expose a secret.
        record["fatal"] = type(error).__name__
    raw = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
    path = output / f"{index:04d}.json"
    with path.open("xb") as stream:
        stream.write(raw)
    return {"source_id": request["source_id"], "file": path.name, "sha256": sha(raw), "fatal": record.get("fatal")}


def capture(requests: list[dict], output: Path, artifact_root: str, code_pin: str) -> dict:
    """Execute only inside the executor's pinned cw-rno2a pilot job."""
    if len(requests) != 22 or len({row["source_id"] for row in requests}) != 22:
        raise ValueError("Only the fixed22-source pilot is supported; no full expansion")
    if not artifact_root.startswith(TEMPORARY_PREFIX):
        raise ValueError("GLM captures require the RNO14-day TTL artifact root")
    if re.fullmatch(r"[0-9a-f]{40}", code_pin) is None:
        raise ValueError("Capture code pin must be a full immutable git revision")
    output.mkdir(parents=True, exist_ok=False)
    manifest_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in requests).encode()
    (output / "request-manifest.jsonl").write_bytes(manifest_bytes)
    base_url = resolve_glm_base_url(RELAY_JOB)
    client = OpenAIChatClient(base_url, os.environ[GLM_BULK_TOKEN_ENV], timeout=900, priority="bulk")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(capture_one, client, row, index, output, base_url) for index, row in enumerate(requests)
        ]
        records = [future.result() for future in futures]
    report = {
        "status": "PILOT_CAPTURE_ONLY_NO_POOL_OR_TRAINING_CLEAR",
        "capture_cluster_binding": CAPTURE_CLUSTER,
        "relay_job": RELAY_JOB,
        "base_url": base_url,
        "requested_model_service": GLM_MODEL,
        "trace_field": TRACE_FIELD,
        "trace_label": "GLM-generated visible rationale; not internal chain-of-thought",
        "code_pin": code_pin,
        "driver_sha256": sha(Path(__file__).read_bytes()),
        "manifest_sha256": sha(manifest_bytes),
        "canonical_off_sha256": CANONICAL_SHA256,
        "context_sha256": CONTEXT_SHA256,
        "artifact_root": artifact_root,
        "request_attempts_per_source": 1,
        "http_timeout_seconds": 900,
        "source_count": len(records),
        "failed_request_count": sum(record["fatal"] is not None for record in records),
        "records": records,
    }
    (output / "capture-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def visible_rationale(response: dict, tokenizer) -> tuple[str, list[int]]:
    choices = response.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("Expected one cleanly stopped GLM response choice")
    message = choices[0].get("message", {})
    text = message.get("content")
    if message.get("tool_calls") or message.get("function_call") or not isinstance(text, str) or not text:
        raise ValueError("Expected text-only visible rationale, without tool output")
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not 8 <= len(ids) <= 96 or set(ids).intersection(SPECIAL_TOKEN_IDS):
        raise ValueError("Visible rationale must contain8-96 lexical native tokens and no protocol controls")
    if tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != text:
        raise ValueError("Visible rationale text/native-ID roundtrip changed text")
    return text, ids


def audit(requests: list[dict], canonical: Path, captures: Path, tokenizer_dir: Path, output: Path) -> dict:
    """Write all22 actual triples and surface/native drafts, pending semantic review."""
    sources = {row["source_id"]: row for row in canonical_sources(canonical)}
    tokenizer = checked_tokenizer(tokenizer_dir)
    expected_files = {f"{index:04d}.json" for index in range(len(requests))}
    actual_files = {path.name for path in captures.glob("[0-9][0-9][0-9][0-9].json")}
    if actual_files != expected_files:
        raise ValueError("Pilot captures missing/extra; never retry a missing source")
    output.mkdir(parents=True, exist_ok=False)
    records, drafts, hashes = [], [], {}
    for index, request in enumerate(requests):
        path = captures / f"{index:04d}.json"
        raw = path.read_bytes()
        hashes[path.name] = sha(raw)
        capture_row = json.loads(raw)
        if (
            capture_row["request"] != request
            or capture_row["relay_job"] != RELAY_JOB
            or capture_row["trace_field"] != TRACE_FIELD
        ):
            raise ValueError("Captured request/relay/trace-field differs from the bound pilot")
        off = sources[request["source_id"]]
        record = {
            "source_id": off["source_id"],
            "family": off["family"],
            "capture": path.name,
            "context": off["messages"][:-1],
            "canonical_final": off["messages"][-1]["content"],
            "failed_trace_policy": "retain exact B-base empty-span ON row",
            "rejection_reasons": [],
        }
        if capture_row.get("fatal"):
            record["rejection_reasons"] = ["request_failure"]
        else:
            try:
                text, ids = visible_rationale(capture_row["response"], tokenizer)
                record["visible_rationale"] = text
                reasons = identity_qc(text, "", record["canonical_final"])
                record["rejection_reasons"] = reasons
                if not reasons:
                    draft = graft_reasoning(off, ids, tokenizer)
                    drafts.append(draft)
                    record.update(
                        status="SURFACE_AND_NATIVE_GRAFT_PASS_PENDING_ALL22_SEMANTIC_READ",
                        final_text_tokens=draft["final_text_tokens"],
                        reasoning_text_tokens=draft["reasoning_text_tokens"],
                    )
            except ValueError as error:
                record["rejection_reasons"] = [str(error)]
        records.append(record)
    (output / "review.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    (output / "draft-on-rows.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in drafts))
    report = {
        "status": "SURFACE_QC_ONLY_ALL22_ACTUAL_TRIPLES_REQUIRE_MARTEN_SEMANTIC_READ",
        "acceptance_denominator_all_preselected_requests": len(requests),
        "surface_and_native_graft_pass": len(drafts),
        "surface_fraction": len(drafts) / len(requests),
        "below_70_percent_surface_acceptance_alert_before_training": len(drafts) / len(requests) < 0.7,
        "semantic_acceptance": "pending; alert must also use the final semantic acceptance fraction",
        "family_surface_pass_counts": dict(Counter(row["family"] for row in drafts)),
        "rejections": dict(Counter(reason for row in records for reason in row["rejection_reasons"])),
        "canonical_final_lexical_tokens_unchanged_all_sources": sum(
            row["final_text_tokens"] for row in sources.values()
        ),
        "accepted_rationale_lexical_tokens": sum(row["reasoning_text_tokens"] for row in drafts),
        "capture_sha256": hashes,
        "comparison": "identical row/exposure schedule including replay; rejected traces keep B-base empty ON span",
        "note": "Pilot fractions are not full-pool or per-arm actual encountered trace coverage.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "capture", "audit"))
    parser.add_argument("--canonical-off", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact-root")
    parser.add_argument("--code-pin")
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    args = parser.parse_args()
    if args.action == "build":
        rows = pilot_requests(args.canonical_off, args.context)
        with args.manifest.open("x") as stream:
            stream.write("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
        print(json.dumps({"manifest_sha256": sha(args.manifest.read_bytes()), "requests": len(rows)}))
        return
    if not args.manifest_sha256 or args.output is None:
        parser.error("capture/audit require --manifest-sha256 and --output")
    rows = checked_manifest(args.manifest, args.manifest_sha256, args.canonical_off, args.context)
    if args.action == "capture":
        if not args.artifact_root or not args.code_pin:
            parser.error("capture requires --artifact-root and --code-pin")
        report = capture(rows, args.output, args.artifact_root, args.code_pin)
    else:
        if args.captures is None or args.tokenizer_dir is None:
            parser.error("audit requires --captures and --tokenizer-dir")
        report = audit(rows, args.canonical_off, args.captures, args.tokenizer_dir, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
