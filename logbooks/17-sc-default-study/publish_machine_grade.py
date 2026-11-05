# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish a captured original machine grade after checking its frozen inputs."""

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq
from rigging.filesystem.buckets import filesystem_for
from taskcompendium.models import ScriptGrader, TaskSpec
from taskcompendium.pipeline.fingerprints import deduplication_key

from experiments.post_training.sc_default_data import write_json
from experiments.post_training.sc_default_regrade import fingerprint, read_json, save_local


def approved_prime_format_failure(task: TaskSpec, verdict: dict, policy: dict) -> bool:
    """Recognize only the approved integer-factor parser rejection."""
    if not policy["approved"] or policy["handler"] != "reasoning-gym-prime-factor-int-value-error-v1":
        return False
    if not isinstance(task.grader, ScriptGrader) or task.grader.environment.packages_lock != policy["packages_lock"]:
        return False
    config = next((r for r in task.resources.verifier if r.path == "config.json"), None)
    if config is None:
        return False
    contract = json.loads(base64.b64decode(config.source.content_base64))["contract"]
    diagnostic = verdict["diagnostics"]
    stderr = diagnostic.get("stderr", "").rstrip().splitlines()
    return (
        contract.get("task") == "prime_factorization"
        and "separated by \u00d7" in contract["entry"]["question"]
        and verdict["status"] == "infra_error"
        and verdict["reward"] is None
        and diagnostic.get("exit_code") == 1
        and "reasoning_gym/arithmetic/prime_factorization.py" in diagnostic.get("stderr", "")
        and "in _normalize_answer" in diagnostic.get("stderr", "")
        and bool(stderr)
        and stderr[-1].startswith("ValueError: invalid literal for int() with base 10:")
    )


def publish(plan_path: Path, tasks_path: Path, stage: Path, format_policy_path: Path | None = None) -> dict:
    """Adopt the first valid recorded verdict without another grading call."""
    plan = json.loads(plan_path.read_text())
    fs, key = filesystem_for(plan["output"])
    raw_result = fs.cat_file(key)
    result = json.loads(raw_result)
    for field in ("export", "step", "uid", "backend", "validation_sha256", "packages_lock", "packages_lock_sha256"):
        if result[field] != plan[field]:
            raise ValueError(f"Machine replay differs from the published plan: {field}")
    if result["source_sha256"] != plan["driver_sha256"] or result["canonical_grade_calls"] != 1:
        raise ValueError("Expected exactly one call from the published replay driver")
    if result.get("qualification_only"):
        raise ValueError("A qualification control cannot replace an existing valid verdict")
    if result.get("resource_guard_recovery") != plan.get("resource_guard_recovery"):
        raise ValueError("Resource-window recovery differs from the published plan")
    if hashlib.sha256(tasks_path.read_bytes()).hexdigest() != result["validation_sha256"]:
        raise ValueError("Local frozen validation differs from the machine input")
    uid = result["uid"]
    task_raw = pq.read_table(tasks_path, columns=["task_spec"]).column(0)[int(uid)].as_py()
    task = TaskSpec.model_validate_json(task_raw)
    if hashlib.sha256(task_raw.encode()).hexdigest() != result["task_spec_sha256"]:
        raise ValueError("Machine replay used another TaskSpec")
    if deduplication_key(task) != result["contract"] or task.id != result["task"]:
        raise ValueError("Machine replay used another grading contract")
    grader = task.grader
    if not isinstance(grader, ScriptGrader) or grader.environment is None:
        raise ValueError("Expected an original external machine grader")
    if result["backend"] != "local" or grader.environment.packages_lock != result["packages_lock"]:
        raise ValueError("Machine backend or original package lock differs")
    if grader.timeout != plan["original_script_timeout"]:
        raise ValueError("Original script timeout differs")
    lock_fs, lock_key = filesystem_for(result["packages_lock"])
    if hashlib.sha256(lock_fs.cat_file(lock_key)).hexdigest() != result["packages_lock_sha256"]:
        raise ValueError("Original package lock bytes changed")
    verdict = result["verdict"]
    policy = json.loads(format_policy_path.read_text()) if format_policy_path is not None else None
    approved_format = policy is not None and approved_prime_format_failure(task, verdict, policy)
    if not approved_format and (verdict["status"] != "graded" or verdict["reward"] is None):
        raise ValueError("A machine error cannot become a task failure or a valid grade")
    reward = 0.0 if approved_format else verdict["reward"]
    if not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError("Invalid original grader reward")
    if (verdict["score_min"], verdict["score_max"]) != (0, 1):
        raise ValueError("Original grader score range differs")
    export = result["export"]
    fs, prefix = filesystem_for(export + f"/dumped_evals/global_step_{result['step']}_evals")
    matches = []
    for key in sorted(fs.find(prefix)):
        if key.endswith(".jsonl") and not key.endswith("/aggregated_results.jsonl"):
            matches.extend(row for line in fs.cat_file(key).splitlines() if str((row := json.loads(line))["uid"]) == uid)
    (row,) = matches
    if fingerprint(row) != result["row_sha256"] or not row["grading_messages"]:
        raise ValueError("Original saved grading conversation changed")
    if (row.get("verification") or {}).get("status") == "verified":
        raise ValueError("A supplemental grade must not replace an original valid verdict")
    recovered = {
        "task": task.id,
        "contract": result["contract"],
        "row_sha256": result["row_sha256"],
        "recovery_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "machine_recovery": {
            **{name: result[name] for name in ("backend", "packages_lock", "packages_lock_sha256", "runtime_identity")},
            "replay_source_sha256": result["source_sha256"],
            "diagnostic_uri": plan["output"],
            "diagnostic_sha256": hashlib.sha256(raw_result).hexdigest(),
            "resource_guard_recovery": result.get("resource_guard_recovery"),
        },
        "original_verification": row.get("verification"),
        "attempts": [{"seconds": result["seconds"], "verdict": verdict}],
        "answer_reconstruction": "saved original grading messages",
        "verification": {
            "status": "verified",
            "score": reward,
            "score_min": 0.0,
            "score_max": 1.0,
            "passed": verdict["passed"],
            "reason": None,
            "diagnostics": {"grade_only_recovery": True, "original_machine_replay": True},
        },
    }
    if approved_format:
        recovered["approved_format_failure"] = {
            "handler": policy["handler"],
            "approval_observed_at_utc": policy["approval_observed_at_utc"],
            "policy_sha256": hashlib.sha256(format_policy_path.read_bytes()).hexdigest(),
        }
    staged = {"export": export, "step": result["step"], "rows": {uid: recovered}}
    if stage.exists() and json.loads(stage.read_text()) != staged:
        raise ValueError("Do not replace a different staged machine verdict")
    save_local(stage, staged)
    uri = export + f"/study/regrades-{result['step']:04d}.json"
    remote_fs, remote_key = filesystem_for(uri)
    existing = read_json(uri) if remote_fs.exists(remote_key) else {"step": result["step"], "rows": {}}
    if existing["step"] != result["step"] or existing.get("export", export) != export:
        raise ValueError("Existing recovery manifest belongs to another run or step")
    if uid in existing["rows"] and existing["rows"][uid] != recovered:
        raise ValueError("Do not replace an existing recovered verdict")
    existing["rows"][uid] = recovered
    write_json(uri, existing)
    return {
        "uid": uid,
        "reward": reward,
        "additional_grading_calls": 0,
        "preserved_other_rows": len(existing["rows"]) - 1,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--format-policy", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.plan, args.tasks, args.stage, args.format_policy)))
