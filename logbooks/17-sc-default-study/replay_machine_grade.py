# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay one saved answer with its original local machine grader."""

import argparse
import hashlib
import io
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import fsspec
import pyarrow.parquet as pq
import yaml
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from taskcompendium.chat import chat_conversation
from taskcompendium.grading_result import Outcome
from taskcompendium.models import GradingAttempt, ScriptGrader, TaskSpec, VerifyitGrader, grades_with_judge
from taskcompendium.pipeline.fingerprints import deduplication_key
from taskcompendium.runtime.local import LocalGraderMachines, local_runtime
from taskcompendium.runtime.task_grading import grade_task

from experiments.post_training.sc_default_data import write_json
from experiments.post_training.sc_default_regrade import fingerprint, save_local


def replay(export: str, tasks: str, step: int, uid: str, output: str) -> dict:
    """Save exactly one canonical grade attempt before any supplemental publication."""
    stage = Path(os.environ["IRIS_OUTPUT_DIR"]) / f"grade-{step:04d}-{uid}.json"
    output_fs, output_key = filesystem_for(output)
    if stage.exists() or output_fs.exists(output_key):
        raise ValueError("A replay result already exists; inspect that saved attempt")
    config_fs, config_key = filesystem_for(export.removesuffix("/exports") + "/resolved-launch.yaml")
    sessions = yaml.safe_load(config_fs.cat_file(config_key))["config"]["skyrl"]["environment"]["task_sessions"]
    if sessions.get("local") is None or sessions.get("daytona"):
        raise ValueError("Recovery supports the original local-only grading backend")
    task_fs, task_key = filesystem_for(tasks)
    task_bytes = task_fs.cat_file(task_key)
    task_raw = pq.read_table(io.BytesIO(task_bytes), columns=["task_spec"]).column(0)[int(uid)].as_py()
    task = TaskSpec.model_validate_json(task_raw)
    if (
        not isinstance(task.grader, ScriptGrader | VerifyitGrader)
        or grades_with_judge(task.grader)
        or task.grader.environment is None
    ):
        raise ValueError("Use the existing judge recovery for an in-process grader")
    fs, prefix = filesystem_for(export + f"/dumped_evals/global_step_{step}_evals")
    matches = []
    for key in sorted(fs.find(prefix)):
        if key.endswith(".jsonl") and not key.endswith("/aggregated_results.jsonl"):
            matches.extend(row for line in fs.cat_file(key).splitlines() if str((row := json.loads(line))["uid"]) == uid)
    (row,) = matches
    if (row.get("verification") or {}).get("status") == "verified":
        raise ValueError("Recovery must not replace a valid verdict")
    messages = row["grading_messages"]
    if not messages:
        raise ValueError("Machine replay requires the exact saved grading conversation")
    lock = task.grader.environment.packages_lock
    if lock is None:
        raise ValueError("Machine replay requires the original pinned package lock")
    lock_fs, _ = filesystem_for(lock)
    configure_coreweave_s3()
    fsspec.config.conf["s3"]["client_kwargs"]["region_name"] = lock_fs.client_kwargs["region_name"]
    runtime = local_runtime(lock)
    factory, spec = LocalGraderMachines().machine(task.grader.environment, memory_mb=1024)
    attempt = GradingAttempt(chat_conversation(messages), stop_reason=row["stop_reason"])
    start = time.perf_counter()
    grade = grade_task(task, attempt, machine_factory=factory, machine_spec=spec)
    result = {
        "export": export,
        "step": step,
        "uid": uid,
        "task": task.id,
        "contract": deduplication_key(task),
        "row_sha256": fingerprint(row),
        "validation_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "task_spec_sha256": hashlib.sha256(task_raw.encode()).hexdigest(),
        "backend": factory.backend.value,
        "packages_lock": lock,
        "packages_lock_sha256": runtime.lock_sha256,
        "runtime_identity": runtime.identity,
        "seconds": time.perf_counter() - start,
        "verdict": asdict(grade),
        "canonical_grade_calls": 1,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "supplemental_grade_published": False,
    }
    save_local(stage, result)
    write_json(output, result)
    if grade.status is not Outcome.GRADED or grade.reward is None:
        raise RuntimeError(f"Saved machine replay still lacks a grade for uid {uid}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export")
    parser.add_argument("tasks")
    parser.add_argument("output")
    parser.add_argument("--step", type=int, required=True, choices=[0, 25, 50])
    parser.add_argument("--uid", required=True)
    args = parser.parse_args()
    result = replay(args.export, args.tasks, args.step, args.uid, args.output)
    print(json.dumps({key: result[key] for key in ["step", "uid", "backend", "canonical_grade_calls"]}))
