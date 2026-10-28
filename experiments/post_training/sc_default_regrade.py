# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover missing judge verdicts from saved answers, without model generation."""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from rigging.filesystem.buckets import filesystem_for
from rigging.secrets import require_secret
from taskcompendium.chat import chat_conversation
from taskcompendium.grading_result import Outcome
from taskcompendium.models import ConversationTrace, GradingAttempt, TaskSpec, TextMessage, grades_with_judge
from taskcompendium.pipeline.fingerprints import deduplication_key
from taskcompendium.runtime.task_grading import grade_task
from transformers import AutoTokenizer
from verifyit.modes.grade_judge import JudgeConnection

from experiments.post_training.sc_default_data import write_json


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(uri: str):
    filesystem, key = filesystem_for(uri)
    with filesystem.open(key, "rt") as stream:
        return json.load(stream)


def regrade(export: str, task_path: Path, step: int, adopted: Path | None = None) -> dict:
    fs, prefix = filesystem_for(export + f"/dumped_evals/global_step_{step}_evals")
    task_specs = pq.read_table(task_path, columns=["task_spec"]).column(0).to_pylist()
    manifest_uri = export + f"/study/regrades-{step:04d}.json"
    recovery_fs, recovery_key = filesystem_for(manifest_uri)
    recovery = read_json(manifest_uri) if recovery_fs.exists(recovery_key) else {"step": step, "rows": {}}
    source_fs, source_key = filesystem_for(export.removesuffix("/exports") + "/resolved-launch.yaml")
    launch = yaml.safe_load(source_fs.cat_file(source_key))["config"]
    judge_model = launch["runtime"]["task_env"]["VERIFYIT_JUDGE_MODEL"]
    os.environ["VERIFYIT_JUDGE_MODEL"] = judge_model
    remote = launch["skyrl"]["environment"]["task_sessions"]["remote_judge"]
    connection = JudgeConnection(remote["base_url"], require_secret(remote["api_key_env"]))
    tokenizer = None
    adopted_grade = json.loads(adopted.read_text()) if adopted is not None else None
    for key in sorted(fs.find(prefix)):
        if not key.endswith(".jsonl") or key.endswith("/aggregated_results.jsonl"):
            continue
        with fs.open(key, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                if (row.get("verification") or {}).get("status") == "verified":
                    continue
                uid = str(row["uid"])
                if uid in recovery["rows"]:
                    if recovery["rows"][uid]["row_sha256"] != fingerprint(row):
                        raise ValueError("Previously recovered answer changed")
                    continue
                raw = task_specs[int(uid)]
                task = TaskSpec.model_validate_json(raw)
                if not grades_with_judge(task.grader):
                    raise ValueError(f"Non-judge failure at uid {uid} requires its original machine recovery")
                messages = row.get("grading_messages")
                reconstruction = "saved original grading messages"
                if messages is not None:
                    attempt = GradingAttempt(chat_conversation(messages), stop_reason=row["stop_reason"])
                else:
                    # Compatibility with the failed initial run before exact
                    # failed-attempt messages were added to the research dump.
                    if row["env_class"] != "taskcompendium" or task.context.events[-1].role == "assistant":
                        raise ValueError("This historical answer needs its original conversation evidence")
                    if tokenizer is None:
                        model = launch["inputs"]["model"]
                        tokenizer = AutoTokenizer.from_pretrained(
                            model["tokenizer_uri"], revision=model["tokenizer_revision"], local_files_only=True
                        )
                    answer = tokenizer.decode(row["response_token_ids"], skip_special_tokens=True)
                    reconstruction = "saved token IDs decoded with the serving API skip_special_tokens invariant"
                    attempt = GradingAttempt(
                        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=answer))),
                        stop_reason=row["stop_reason"],
                    )
                attempts = []
                if adopted_grade is not None and adopted_grade["uid"] == uid:
                    if adopted_grade["task_spec_sha256"] != hashlib.sha256(raw.encode()).hexdigest():
                        raise ValueError("Adopted recovery uses a different task contract")
                    if (
                        adopted_grade["response_token_ids_sha256"]
                        != hashlib.sha256(json.dumps(row["response_token_ids"]).encode()).hexdigest()
                    ):
                        raise ValueError("Adopted recovery uses a different saved answer")
                    if (
                        messages is None
                        and adopted_grade["answer_sha256"] != hashlib.sha256(answer.encode()).hexdigest()
                    ):
                        raise ValueError("Adopted recovery reconstructed different answer bytes")
                    verdict = adopted_grade["regrade"]
                    if verdict["status"] != Outcome.GRADED.value:
                        raise ValueError("Only the first valid already-recorded verdict may be adopted")
                    attempts.append({"adopted_proof_sha256": hashlib.sha256(adopted.read_bytes()).hexdigest()})
                else:
                    for _ in range(3):
                        started = time.perf_counter()
                        grade = grade_task(task, attempt, judge_connection=connection)
                        verdict = asdict(grade)
                        attempts.append({"seconds": time.perf_counter() - started, "verdict": verdict})
                        if grade.status is not Outcome.INFRA_ERROR:
                            break
                if verdict["status"] != Outcome.GRADED or verdict["reward"] is None:
                    write_json(manifest_uri + ".pending", {"uid": uid, "attempts": attempts})
                    raise RuntimeError(f"Judge recovery still lacks a verdict at uid {uid}")
                recovery["rows"][uid] = {
                    "task": task.id,
                    "contract": deduplication_key(task),
                    "row_sha256": fingerprint(row),
                    "judge_model": judge_model,
                    "judge_base_url": remote["base_url"],
                    "original_verification": row.get("verification"),
                    "attempts": attempts,
                    "answer_reconstruction": reconstruction,
                    "verification": {
                        "status": "verified",
                        "score": verdict["reward"],
                        "score_min": 0.0,
                        "score_max": 1.0,
                        "passed": None,
                        "reason": None,
                        "diagnostics": {"grade_only_recovery": True},
                    },
                }
                write_json(manifest_uri, recovery)
    return recovery


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export")
    parser.add_argument("tasks", type=Path)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument(
        "--adopt", type=Path, help="Keep the first valid verdict already recorded during operator recovery"
    )
    args = parser.parse_args()
    print(json.dumps(regrade(args.export.rstrip("/"), args.tasks, args.step, args.adopt), indent=2))
