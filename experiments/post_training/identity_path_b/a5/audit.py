# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""All-source raw joins and fixed QC, before any A5 packed training admission."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.post_training.identity_path_b.a5.qc import checked_student


def audit(rows: list[dict], captures: Path, tokenizer, output: Path) -> dict:
    by_index = {row["global_index"]: row for row in rows}
    if len(by_index) != len(rows):
        raise ValueError("Repeated A5 manifest original")
    output.mkdir(parents=True, exist_ok=False)
    seen, review, drafts, hashes = set(), [], [], {}
    for path in sorted(captures.glob("[0-9]*.json")):
        raw = path.read_bytes()
        record = json.loads(raw)
        index = record["index"]
        if index in seen or record["candidate"] != by_index[index]:
            raise ValueError("Raw request differs from frozen original/teacher/seed/context")
        seen.add(index)
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        candidate = record["candidate"]
        draft, flags = checked_student(record, tokenizer)
        review.append(
            dict(
                source_id=candidate["source_id"],
                family=candidate["family"],
                flags=flags,
                capture=path.name,
                candidate_pass=draft is not None,
            )
        )
        if draft is not None:
            drafts.append(draft)
    for index in sorted(set(by_index) - seen):
        row = by_index[index]
        review.append(
            dict(source_id=row["source_id"], family=row["family"], flags=["not_attempted"], candidate_pass=False)
        )
    (output / "review.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in review))
    (output / "draft-ON.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in drafts))
    report = dict(
        status="PRESCREEN_ONLY_PENDING_ROLE_AWARE_ADMISSION_NO_TRAINING_CLEAR",
        sources=len(rows),
        attempted=len(seen),
        surface_native_pass=len(drafts),
        yield_fraction=len(drafts) / len(rows),
        alert_below_25_percent=len(drafts) * 4 < len(rows),
        source_families=dict(Counter(row["family"] for row in rows)),
        pass_families=dict(Counter(row["family"] for row in drafts)),
        rejection_flags=dict(Counter(flag for row in review for flag in row["flags"])),
        pass_final_text_tokens=sum(row["final_text_tokens"] for row in drafts),
        pass_reasoning_text_tokens=sum(row["reasoning_text_tokens"] for row in drafts),
        raw_sha256=hashes,
        policy=(
            "Original OFF unchanged; rejected ON dropped/no fallback/no retry/no rebalancing; "
            "coupled teacher-ON replacement"
        ),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
