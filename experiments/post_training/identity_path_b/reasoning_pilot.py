# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build or capture exactly22 fixed, distinct ON identity-teacher pilot requests."""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from experiments.post_training.identity_path_b.reasoning import (
    IDENTITY_POOL_SHA256,
    TEACHER_CONTEXT,
    graft_reasoning,
    identity_qc,
    teacher_candidate,
)
from experiments.post_training.identity_t2.capture import (
    checked_tokenizer,
    echoes_capability,
    generated_ids,
    prompt_ids,
    request_body,
    roundtrip,
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def pilot_sources(rows: list[dict]) -> list[dict]:
    """Cover11 families, all8 denial targets, plus project/multiturn contexts."""
    ordered = sorted(rows, key=lambda row: sha(f"310:reason-pilot:{row['source_id']}".encode()))
    chosen = {}
    families, denials = set(), set()
    for row in ordered:
        answer = row["normalized_answer"]
        if row["family"] not in families or (row["family"] == "false_premise" and answer not in denials):
            chosen[row["source_id"]] = row
            families.add(row["family"])
            if row["family"] == "false_premise":
                denials.add(answer)
    if len(families) != 11 or len(denials) != 8 or len(chosen) != 18:
        raise ValueError("Expected11 families with8 distinct lab-denial finals")
    for family in ("project_fact", "name", "unknown_fact", "open_weight_data"):
        extra = next(
            row
            for row in ordered
            if row["source_id"] not in chosen
            and row["family"] == family
            and (family == "project_fact" or len(row["messages"]) > 2)
        )
        chosen[extra["source_id"]] = extra
    return list(chosen.values())


def build(pool: Path, output: Path) -> dict:
    if sha(pool.read_bytes()) != IDENTITY_POOL_SHA256:
        raise ValueError("Reviewed native R4 identity pool changed")
    rows = []
    with pool.open() as stream:
        for line in stream:
            row = json.loads(line)
            if not row["enable_thinking"]:
                rows.append(row)
    selected = pilot_sources(rows)
    output.mkdir(exist_ok=False, parents=True)
    candidates = [teacher_candidate(row) for row in selected]
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidates))
    (output / "canonical-off.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    report = {
        "status": "PILOT_PROPOSAL_NO_GENERATION_CLEAR",
        "requests": len(candidates),
        "distinct_original_sources": len({row["source_id"] for row in candidates}),
        "mode": "ON only; one sample/source; zero retries",
        "family_counts": dict(Counter(row["family"] for row in selected)),
        "distinct_denial_targets": len(
            {row["normalized_answer"] for row in selected if row["family"] == "false_premise"}
        ),
        "multiturn_sources": sum(len(row["messages"]) > 2 for row in selected),
        "canonical_final_lexical_tokens": sum(row["final_text_tokens"] for row in selected),
        "pool_sha256": IDENTITY_POOL_SHA256,
        "manifest_sha256": sha(manifest.read_bytes()),
        "canonical_off_sha256": sha((output / "canonical-off.jsonl").read_bytes()),
        "shared_teacher_context_sha256": sha(TEACHER_CONTEXT.encode()),
        "teacher_system_sha256_by_source": {row["source_id"]: sha(row["teacher_system"].encode()) for row in candidates},
        "driver_sha256": sha(Path(__file__).read_bytes()),
        "reasoning_module_sha256": sha(Path(__file__).with_name("reasoning.py").read_bytes()),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def capture_one(index: int, row: dict, *, endpoint: str, model: str, tokenizer, output: Path) -> dict:
    """Reuse T2 request/token proofs; explicitly lower this teacher's output budget."""
    body = {**request_body(row, model), "max_tokens": row["max_tokens"]}
    record = {"index": index, "candidate": row, "request": body}
    with (output / f"{index:04d}.json").open("x") as file:
        try:
            cpu_prefix = prompt_ids(tokenizer, row, teacher=True)
            tokenize_body = {key: body[key] for key in ("model", "messages", "chat_template_kwargs")}
            tokenize_body["add_generation_prompt"] = True
            response = requests.post(f"{endpoint.removesuffix('/v1')}/tokenize", json=tokenize_body, timeout=(15, 60))
            response.raise_for_status()
            server_prefix = response.json()["tokens"]
            record.update(cpu_teacher_prompt_token_ids=cpu_prefix, server_teacher_prompt_token_ids=server_prefix)
            if server_prefix != cpu_prefix:
                raise ValueError("CPU/server teacher prompt mismatch; no completion requested")
            response = requests.post(f"{endpoint}/chat/completions", json=body, timeout=(15, 300))
            record["http_status"] = response.status_code
            if echoes_capability(response.text, endpoint):
                raise ValueError("Sensitive endpoint echoed; payload not serialized")
            response.raise_for_status()
            payload = response.json()
            ids = generated_ids(payload)
            record.update(
                response=payload,
                raw_token_ids=ids,
                roundtrip=roundtrip(tokenizer, row, ids),
                student_prompt_token_ids=prompt_ids(tokenizer, row),
                finish_reason=payload["choices"][0].get("finish_reason"),
            )
        except Exception as error:
            # Transport exceptions can contain endpoint capabilities.
            record.update(error_type=type(error).__name__, fatal="capture_failure_no_retry")
        json.dump(record, file, ensure_ascii=False)
        file.write("\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("build")
    prepare.add_argument("--pool", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--manifest", type=Path, required=True)
    capture.add_argument("--manifest-sha256", required=True)
    capture.add_argument("--tokenizer-dir", type=Path, required=True)
    capture.add_argument("--endpoint", required=True)
    capture.add_argument("--model", required=True)
    capture.add_argument("--output", type=Path, required=True)
    proof = commands.add_parser("prove")
    proof.add_argument("--canonical-off", type=Path, required=True)
    proof.add_argument("--tokenizer-dir", type=Path, required=True)
    proof.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build(args.pool, args.output), indent=2))
        return
    if args.command == "prove":
        tokenizer = checked_tokenizer(args.tokenizer_dir)
        fixture = (
            "I am Marin's Latest MoE. The Marin Community trained me; "
            "I should answer the actual question without guessing."
        )
        fixture_ids = tokenizer.encode(fixture, add_special_tokens=False)
        rows = [json.loads(line) for line in args.canonical_off.read_text().splitlines()]
        records = []
        for row in rows:
            result = graft_reasoning(row, fixture_ids, tokenizer)
            if identity_qc(fixture, row["messages"][-1]["content"], row["messages"][-1]["content"]):
                raise ValueError("QC fixture or canonical profile failed")
            records.append(
                {
                    "source_id": row["source_id"],
                    "family": row["family"],
                    "final_text_tokens": result["final_text_tokens"],
                    "reasoning_text_tokens": result["reasoning_text_tokens"],
                    "full_id_sha256": sha(json.dumps(result["input_ids"]).encode()),
                    "prefix_and_full_and_raw_reasoning_and_canonical_final_ids": "PASS",
                }
            )
        args.output.write_text(
            json.dumps(
                {
                    "status": "CPU_SYNTHETIC_FIXTURE_ONLY_NOT_TRAINING_REASONING",
                    "tokenizer_json_sha256": sha((args.tokenizer_dir / "tokenizer.json").read_bytes()),
                    "template_sha256": sha((args.tokenizer_dir / "chat_template.jinja").read_bytes()),
                    "canonical_off_sha256": sha(args.canonical_off.read_bytes()),
                    "fixture_text": fixture,
                    "fixture_ids": fixture_ids,
                    "records": records,
                },
                indent=2,
            )
            + "\n"
        )
        return
    if sha(args.manifest.read_bytes()) != args.manifest_sha256:
        raise ValueError("Pilot manifest changed")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    if len(rows) != 22 or len({row["source_id"] for row in rows}) != 22:
        raise ValueError("Pilot requires exactly22 distinct original sources")
    if any(not row["enable_thinking"] or row["max_tokens"] != 256 for row in rows):
        raise ValueError("Pilot mode/budget differs from reviewed request")
    tokenizer = checked_tokenizer(args.tokenizer_dir)
    args.output.mkdir(exist_ok=False, parents=True)
    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(
            workers.map(
                lambda item: capture_one(
                    item[0], item[1], endpoint=args.endpoint, model=args.model, tokenizer=tokenizer, output=args.output
                ),
                enumerate(rows),
            )
        )
    (args.output / "capture-summary.json").write_text(
        json.dumps(
            {
                "requests": len(results),
                "fatal": sum(bool(row.get("fatal")) for row in results),
                "manifest_sha256": args.manifest_sha256,
                "retries": 0,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
