# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay one saved answer with its original local machine grader."""

import argparse
import base64
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
from skyrl_gym.envs.lcb.livecodebench import extract_code_from_model, normalize_lcb_ground_truth
from taskcompendium.chat import chat_conversation
from taskcompendium.grading_result import Outcome
from taskcompendium.models import GradingAttempt, InlineFile, ScriptGrader, TaskSpec, VerifyitGrader, grades_with_judge
from taskcompendium.pipeline.fingerprints import deduplication_key
from taskcompendium.runtime.local import LocalGraderMachines, local_runtime
from taskcompendium.runtime.task_grading import grade_task
from taskcompendium.submission import conversation_messages

from experiments.post_training.sc_default_data import write_json
from experiments.post_training.sc_default_regrade import fingerprint, save_local


def full_lcb_test_window(task: TaskSpec) -> tuple[TaskSpec, dict]:
    """Retain the original tests and per-test limits, using their finite full-suite window."""
    grader = task.grader
    if not isinstance(grader, ScriptGrader) or grader.argv != ("python3", "/tests/grade.py"):
        raise ValueError("Expected the original released competitive-code script")
    resources = {r.path: base64.b64decode(r.source.content_base64) for r in task.resources.verifier}
    config = json.loads(resources["config.json"])["contract"]
    tests = json.loads(normalize_lcb_ground_truth(config["verifier_metadata"]["unit_tests"]))
    code_source = resources["skyrl_gym/envs/nemotron_ultra/code_gen.py"].decode()
    lcb_source = resources["skyrl_gym/envs/lcb/livecodebench.py"].decode()
    if (
        "DEFAULT_PER_TEST_TIMEOUT_SECONDS = 10" not in code_source
        or "deadline = (timeout + 1) * num_tests + 5" not in lcb_source
    ):
        raise ValueError("The released scorer's per-test window differs")
    script = resources["grade.py"].decode()
    original_import = "from skyrl_gym.envs.nemotron_ultra.code_gen import grade_code"
    original_call = 'grade_code(message["content"], contract, assistant_message=message)'
    if script.count(original_import) != 1 or script.count(original_call) != 1:
        raise ValueError("Unrecognized original competitive-code entrypoint")
    updated = script.replace(
        original_import, "from skyrl_gym.envs.lcb.livecodebench import VerifierLimits\n" + original_import
    )
    updated = updated.replace(original_call, original_call[:-1] + ", limits=VerifierLimits(total_timeout_seconds=None))")
    window = (10 + 1) * len(tests) + 5
    script_timeout = max(grader.timeout, window + 60)
    changed = tuple(
        (
            r.model_copy(update={"source": InlineFile(content_base64=base64.b64encode(updated.encode()).decode())})
            if r.path == "grade.py"
            else r
        )
        for r in task.resources.verifier
    )
    replay_task = task.model_copy(
        update={
            "grader": grader.model_copy(update={"timeout": script_timeout}),
            "resources": task.resources.model_copy(update={"verifier": changed}),
        }
    )
    return replay_task, {
        "tests": len(tests),
        "per_test_timeout_seconds": 10,
        "finite_child_deadline_seconds": window,
        "original_script_timeout_seconds": grader.timeout,
        "replay_script_timeout_seconds": script_timeout,
        "max_memory_bytes": 4 * 1024**3,
        "original_grade_script_sha256": hashlib.sha256(resources["grade.py"]).hexdigest(),
        "replay_grade_script_sha256": hashlib.sha256(updated.encode()).hexdigest(),
        "original_code_scorer_sha256": (
            hashlib.sha256(resources["skyrl_gym/envs/nemotron_ultra/code_gen.py"]).hexdigest()
        ),
        "original_lcb_executor_sha256": hashlib.sha256(resources["skyrl_gym/envs/lcb/livecodebench.py"]).hexdigest(),
        "original_test_config_sha256": hashlib.sha256(resources["config.json"]).hexdigest(),
        "replay_task_spec_sha256": hashlib.sha256(replay_task.model_dump_json().encode()).hexdigest(),
        "replay_contract": deduplication_key(replay_task),
        "scope": (
            "The 300-second process resource cap is replaced by the existing finite sum-of-test windows. "
            "Original test data, candidate code, ordering, short circuit, ten-second per-test limit "
            "and four-GiB child memory cap are retained."
        ),
    }


def replay(
    export: str,
    tasks: str,
    step: int,
    uid: str,
    output: str,
    lcb_full_test_window: bool = False,
    qualification_control: bool = False,
) -> dict:
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
    originally_valid = (row.get("verification") or {}).get("status") == "verified"
    if originally_valid and not qualification_control:
        raise ValueError("Recovery must not replace a valid verdict")
    if qualification_control and not originally_valid:
        raise ValueError("A qualification control requires a previously valid verdict")
    messages = row["grading_messages"]
    reconstruction = "saved original grading messages"
    control_code_sha256 = None
    if not messages and qualification_control:
        code = extract_code_from_model(row["output_response"])
        if code is None:
            raise ValueError("A code control requires a captured fenced program")
        control_code_sha256 = hashlib.sha256(code.encode()).hexdigest()
        messages = [
            *conversation_messages(task.context.events),
            {"role": "assistant", "content": f"```python\n{code}\n```"},
        ]
        reconstruction = (
            "qualification only: captured successful program in a fresh fence; no existing grade is replaced"
        )
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
    replay_task, guard_recovery = full_lcb_test_window(task) if lcb_full_test_window else (task, None)
    start = time.perf_counter()
    grade = grade_task(replay_task, attempt, machine_factory=factory, machine_spec=spec)
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
        "resource_guard_recovery": guard_recovery,
        "qualification_only": qualification_control,
        "original_valid_score": row["verification"]["score"] if originally_valid else None,
        "answer_reconstruction": reconstruction,
        "qualification_control_code_sha256": control_code_sha256,
        "supplemental_grade_published": False,
    }
    save_local(stage, result)
    write_json(output, result)
    if grade.status is not Outcome.GRADED or grade.reward is None:
        raise RuntimeError(f"Saved machine replay still lacks a grade for uid {uid}")
    if qualification_control and grade.reward != row["verification"]["score"]:
        raise RuntimeError("Extended outer resource window changed a valid control verdict")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export")
    parser.add_argument("tasks")
    parser.add_argument("output")
    parser.add_argument("--step", type=int, required=True, choices=[0, 25, 50])
    parser.add_argument("--uid", required=True)
    parser.add_argument("--lcb-full-test-window", action="store_true")
    parser.add_argument("--qualification-control", action="store_true")
    args = parser.parse_args()
    result = replay(
        args.export, args.tasks, args.step, args.uid, args.output, args.lcb_full_test_window, args.qualification_control
    )
    print(json.dumps({key: result[key] for key in ["step", "uid", "backend", "canonical_grade_calls"]}))
