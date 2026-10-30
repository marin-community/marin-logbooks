# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit prefill-aware successor to T2 capture; generated IDs never change."""

import hashlib
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import requests
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.identity_path_b.a7.data import TEACHER, request_body, student_graft, teacher_prompt_ids
from experiments.post_training.identity_path_b.checkpoints import TEMPORARY_PREFIX
from experiments.post_training.identity_t2 import capture as t2


def capture_one(index: int, row: dict[str, Any], *, endpoint: str, model: str, tokenizer: Any, out: Path) -> dict:
    if model != TEACHER:
        raise ValueError("A7 exact teacher model alias changed")
    body = request_body(row)
    record: dict[str, Any] = {
        "index": index,
        "candidate": row,
        "request": body,
        "started_unix": time.time(),
        "cpu_teacher_prompt_token_ids": teacher_prompt_ids(tokenizer, row),
        "student_prompt_token_ids": t2.prompt_ids(tokenizer, row),
    }
    with (out / f"{index:04d}.json").open("x") as file:
        try:
            token_body = {
                key: body[key] for key in ("model", "messages", "tools", "chat_template_kwargs") if key in body
            }
            token_body["add_generation_prompt"] = True
            token_body["continue_final_message"] = False
            record["tokenize_request_issued"] = True
            token_response = requests.post(f"{endpoint.removesuffix('/v1')}/tokenize", json=token_body, timeout=(15, 60))
            if not token_response.ok:
                raise ValueError("Serving tokenizer request failed")
            record["server_teacher_prompt_token_ids"] = token_response.json().get("tokens")
            if record["server_teacher_prompt_token_ids"] != record["cpu_teacher_prompt_token_ids"]:
                raise ValueError("Serving/CPU teacher prompt mismatch; completion not requested")
            record["completion_request_issued"] = True
            response = requests.post(f"{endpoint}/chat/completions", json=body, timeout=(15, 300))
            record["http_status"] = response.status_code
            # Transport exceptions can contain capabilities. Do not serialize their text.
            record["response_body_sha256"] = hashlib.sha256(response.content).hexdigest()
            if t2.echoes_capability(response.text, endpoint):
                raise ValueError("Sensitive endpoint echoed; payload not serialized")
            try:
                payload = response.json()
            except ValueError:
                record["non_json_response_body"] = response.text
                raise ValueError("Non-JSON completion response") from None
            record["response"] = payload
            if not response.ok:
                raise ValueError("Non-success HTTP response")
            record["returned_teacher_prompt_token_ids"] = payload.get("prompt_token_ids")
            if record["returned_teacher_prompt_token_ids"] != record["cpu_teacher_prompt_token_ids"]:
                raise ValueError("Completion provider/native prefixed prompt mismatch")
            ids = t2.generated_ids(payload)
            record["raw_token_ids"] = ids
            record["prefill_student_graft"] = student_graft(tokenizer, row, ids)
            record["length_stop"] = payload["choices"][0].get("finish_reason") == "length"
            if not record["prefill_student_graft"]["ok"] and not record["length_stop"]:
                record["fidelity_rejected"] = True
        except Exception as error:
            record["error_type"] = type(error).__name__
            record["fatal"] = "transport_protocol_or_capture_failure"
        finally:
            record["finished_unix"] = time.time()
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")
    return record


def capture(rows: list[dict], *, endpoint: str, tokenizer, output: Path, artifact_root: str) -> dict:
    if not endpoint.startswith("https://") or not endpoint.endswith("/v1"):
        raise ValueError("Require an owned native TLS endpoint ending/v1")
    if not artifact_root.startswith(TEMPORARY_PREFIX):
        raise ValueError("A7 raw streams require RNO14d")
    remote = StoragePath(artifact_root) / "raw"
    if remote.exists():
        raise FileExistsError("A7 raw stream prefix already exists; no resume or retry")
    output.mkdir(parents=True, exist_ok=False)
    for row in rows:
        teacher_prompt_ids(tokenizer, row)
        t2.prompt_ids(tokenizer, row)
    budget = t2.FidelityBudget()
    records = []
    cursor = 0
    stop = False

    def capture_and_persist(row):
        index = row["global_index"]
        record = capture_one(index, row, endpoint=endpoint, model=TEACHER, tokenizer=tokenizer, out=output)
        raw = (output / f"{index:04d}.json").read_bytes()
        target = remote / f"{index:05d}.json"
        target.write_bytes(raw)
        if target.read_bytes() != raw:
            raise ValueError("Completed A7 raw record remote readback differs")
        receipt = {
            "path": str(target),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "application_writes": 1,
        }
        (output / f"persist-{index:05d}.json").write_text(json.dumps(receipt) + "\n")
        return record

    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = set()
        while cursor < len(rows) or pending:
            while not stop and cursor < len(rows) and len(pending) < 8:
                row = rows[cursor]
                pending.add(pool.submit(capture_and_persist, row))
                cursor += 1
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                record = future.result()
                records.append(record)
                stop = budget.observe(dict(record, roundtrip=record.get("prefill_student_graft", {}))) or stop
    stop = budget.finish() or stop
    report = dict(
        planned=len(rows),
        attempted=len(records),
        unattempted=len(rows) - len(records),
        fidelity_rejects=budget.rejects,
        fidelity_checkpoints=budget.checkpoints,
        fatal_protocol=budget.fatal_protocol,
        stopped=stop,
        one_sample_per_original=True,
        application_retries=0,
        pool_and_training_admission=False,
        completed_record_stream=(
            "Exact raw bytes uploaded and full-readback verified before future completes; " "per-record local receipt"
        ),
    )
    (output / "capture-report.json").write_text(json.dumps(report, indent=2) + "\n")
    if stop:
        raise RuntimeError("A7 shard stopped on protocol failure or >5percent fidelity; no retry")
    return report
