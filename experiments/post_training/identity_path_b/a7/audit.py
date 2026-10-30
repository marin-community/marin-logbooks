# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""All-source raw joins and fixed QC, before any A7 packed training admission."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.post_training.identity_path_b.a5b.data import pilot_gate
from experiments.post_training.identity_path_b.a7.data import teacher_prompt_ids
from experiments.post_training.identity_path_b.a7.qc import checked_student
from experiments.post_training.identity_t2 import capture as t2


def audit(rows: list[dict], captures: Path, tokenizer, output: Path) -> dict:
    by_index = {row["global_index"]: row for row in rows}
    if len(by_index) != len(rows):
        raise ValueError("Repeated A7 manifest original")
    output.mkdir(parents=True, exist_ok=False)
    seen, review, drafts, hashes, records = set(), [], [], {}, []
    for path in sorted(captures.glob("[0-9]*.json")):
        raw = path.read_bytes()
        record = json.loads(raw)
        index = record["index"]
        if index in seen or record["candidate"] != by_index[index]:
            raise ValueError("Raw request differs from frozen original/teacher/seed/context")
        seen.add(index)
        records.append(record)
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        candidate = record["candidate"]
        if record["cpu_teacher_prompt_token_ids"] != teacher_prompt_ids(tokenizer, candidate) or record[
            "student_prompt_token_ids"
        ] != t2.prompt_ids(tokenizer, candidate):
            raise ValueError("Stored CPU/native teacher or student prefix changed")
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
            dict(
                source_id=row["source_id"],
                family=row["family"],
                flags=["no_durable_record_issuance_unknown"],
                candidate_pass=False,
            )
        )
    (output / "review.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in review))
    (output / "draft-ON.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in drafts))
    report = dict(
        status="PRESCREEN_ONLY_PENDING_ROLE_AWARE_ADMISSION_NO_TRAINING_CLEAR",
        sources=len(rows),
        attempted_known=sum(bool(record.get("completion_request_issued")) for record in records),
        surface_native_pass=len(drafts),
        yield_fraction=len(drafts) / len(rows),
        alert_below_25_percent=len(drafts) * 4 < len(rows),
        coverage=dict(
            planned=len(rows),
            durable_records=len(seen),
            missing_durable_issuance_unknown=len(rows) - len(seen),
            completion_requests_issued_known=sum(bool(record.get("completion_request_issued")) for record in records),
            HTTP200=sum(record.get("http_status") == 200 for record in records),
        ),
        fidelity=dict(
            graft_PASS=sum(record.get("prefill_student_graft", {}).get("ok") is True for record in records),
            graft_reject=sum(record.get("prefill_student_graft", {}).get("ok") is False for record in records),
            not_proved=sum("prefill_student_graft" not in record for record in records),
        ),
        durable_records=len(seen),
        missing_durable_records=len(rows) - len(seen),
        capture_fatal=sum("capture_transport_or_protocol_failure" in row["flags"] for row in review),
        model_length=sum("model_length_stop" in row["flags"] for row in review),
        source_families=dict(Counter(row["family"] for row in rows)),
        pass_families=dict(Counter(row["family"] for row in drafts)),
        rejection_flags=dict(Counter(flag for row in review for flag in row["flags"])),
        pass_final_text_tokens=sum(row["final_text_tokens"] for row in drafts),
        pass_reasoning_text_tokens=sum(row["reasoning_text_tokens"] for row in drafts),
        raw_sha256=hashes,
        policy="A7 pilot only; exact raw/native student-format candidates; no pool/full-capture/training authority",
        necessary_gate=pilot_gate(review),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
