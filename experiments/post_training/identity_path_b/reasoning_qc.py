# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit actual pilot raw traces; surface passes remain pending role-aware review."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.post_training.identity_path_b.reasoning import (
    END_THINK,
    graft_reasoning,
    identity_qc,
    reasoning_ids,
    teacher_candidate,
)
from experiments.post_training.identity_t2.capture import checked_tokenizer, generated_ids, prompt_ids, roundtrip


def audit(canonical: Path, captures: Path, tokenizer_dir: Path, output: Path) -> dict:
    tokenizer = checked_tokenizer(tokenizer_dir)
    sources = {row["source_id"]: row for row in map(json.loads, canonical.read_text().splitlines())}
    if len(sources) != 22:
        raise ValueError("Expected exactly22 canonical pilot sources")
    output.mkdir(parents=True, exist_ok=False)
    records, drafts, seen, receipt = [], [], set(), {}
    for path in sorted(captures.glob("[0-9][0-9][0-9][0-9].json")):
        raw = path.read_bytes()
        receipt[path.name] = hashlib.sha256(raw).hexdigest()
        record = json.loads(raw)
        candidate = record["candidate"]
        source = candidate["source_id"]
        if source not in sources or source in seen:
            raise ValueError("Unexpected or repeated original source in pilot capture")
        seen.add(source)
        off = sources[source]
        if candidate != teacher_candidate(off):
            raise ValueError("Teacher capture differs from bound source/context/seed/budget")
        result = {"source_id": source, "family": off["family"], "capture": path.name}
        if record.get("fatal"):
            result["rejection_reasons"] = ["capture_failure"]
            records.append(result)
            continue
        ids = record["raw_token_ids"]
        if ids != generated_ids(record["response"]):
            raise ValueError("Raw target IDs differ from original response")
        teacher_prefix = prompt_ids(tokenizer, candidate, teacher=True)
        if any(
            record[key] != teacher_prefix for key in ("cpu_teacher_prompt_token_ids", "server_teacher_prompt_token_ids")
        ):
            raise ValueError("CPU/server teacher prompt proof differs from native tokenizer")
        if record["student_prompt_token_ids"] != prompt_ids(tokenizer, candidate):
            raise ValueError("Captured student prefix differs from native tokenizer")
        fidelity = roundtrip(tokenizer, candidate, ids)
        if not fidelity["ok"]:
            result["rejection_reasons"] = ["raw_native_roundtrip_failure"]
            records.append(result)
            continue
        try:
            trace_ids = reasoning_ids(ids, record["finish_reason"])
        except ValueError:
            result["rejection_reasons"] = ["invalid_or_overlength_reasoning_span"]
            records.append(result)
            continue
        decode = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        trace = tokenizer.decode(trace_ids, **decode)
        generated_final = tokenizer.decode(ids[ids.index(END_THINK) + 1 : -1], **decode)
        canonical_final = off["messages"][-1]["content"]
        result.update(reasoning=trace, generated_final=generated_final, canonical_final=canonical_final)
        reasons = identity_qc(trace, generated_final, canonical_final)
        result["rejection_reasons"] = reasons
        if reasons:
            records.append(result)
            continue
        draft = graft_reasoning(off, trace_ids, tokenizer)
        result.update(
            status="SURFACE_PASS_NATIVE_GRAFT_PASS_PENDING_ROLE_AWARE_REVIEW",
            final_text_tokens=draft["final_text_tokens"],
            reasoning_text_tokens=draft["reasoning_text_tokens"],
        )
        drafts.append(draft)
        records.append(result)
    if seen != set(sources):
        raise ValueError("Incomplete pilot capture; no missing-source retry is permitted")
    (output / "review.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    (output / "draft-on-rows.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in drafts))
    report = {
        "status": "PILOT_QC_ONLY_NO_POOL_OR_TRAINING_CLEAR",
        "captured_sources": len(seen),
        "surface_and_native_graft_pass_sources": len(drafts),
        "role_review_accepted_sources": "pending",
        "family_surface_pass_counts": dict(Counter(row["family"] for row in drafts)),
        "rejections": dict(Counter(reason for row in records for reason in row["rejection_reasons"])),
        "canonical_final_lexical_pool_before": sum(row["final_text_tokens"] for row in sources.values()),
        "canonical_final_lexical_pool_surface_pass": sum(row["final_text_tokens"] for row in drafts),
        "capture_sha256": receipt,
        "canonical_off_sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(),
        "note": "Pilot yield cannot establish full-pool lexical dose; full accepted pool must be counted independently.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-off", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.canonical_off, args.captures, args.tokenizer_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
