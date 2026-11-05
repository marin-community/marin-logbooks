# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect terminal task snapshots once; this never polls jobs or starts followers."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from prepare_r02_terminal_collection import prepare


def save_new(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def collect_task(task_id, directory, execute):
    digest = hashlib.sha256(task_id.encode()).hexdigest()
    result_path = directory / f"{digest}.json"
    guard_path = directory / f"{digest}.started.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["task_id"] != task_id:
            raise ValueError("Cached terminal task identity mismatch")
        return result
    if guard_path.exists():
        raise ValueError(f"Incomplete prior task read requires reconciliation: {task_id}")
    argv = [
        "uv",
        "run",
        "--no-sync",
        "iris",
        "--cluster=marin",
        "rpc",
        "controller",
        "get-task-status",
        "--task-id",
        task_id,
    ]
    save_new(guard_path, {"task_id": task_id, "argv": argv, "started_utc": datetime.now(UTC).isoformat()})
    response = execute(argv)
    raw_path = directory / f"{digest}.response.json"
    save_new(raw_path, {"returncode": response.returncode, "stdout": response.stdout, "stderr": response.stderr})
    if response.returncode:
        raise RuntimeError(f"Task read failed; response preserved at {raw_path}")
    task = json.loads(response.stdout)["task"]
    if task["task_id"] != task_id:
        raise ValueError("Returned task identity mismatch")
    terminal_states = {
        "TASK_STATE_SUCCEEDED",
        "TASK_STATE_FAILED",
        "TASK_STATE_KILLED",
        "TASK_STATE_WORKER_FAILED",
        "TASK_STATE_UNSCHEDULABLE",
        "TASK_STATE_COSCHED_FAILED",
    }
    terminal_attempt_states = terminal_states | {"TASK_STATE_PREEMPTED"}
    if task["state"] not in terminal_states:
        raise ValueError("Task not terminal; preserve snapshot and reconcile before another read")
    attempts = task["attempts"]
    ids = [a["attempt_id"] for a in attempts]
    uids = [a["attempt_uid"] for a in attempts]
    if not attempts or len(set(ids)) != len(ids) or len(set(uids)) != len(uids) or any(not uid for uid in uids):
        raise ValueError("Missing or duplicate attempt identity")
    if task["current_attempt_id"] not in ids or any(a["state"] not in terminal_attempt_states for a in attempts):
        raise ValueError("Incomplete terminal attempt reconciliation")
    result = {
        "task_id": task_id,
        "task": task,
        "raw_evidence": str(raw_path),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "collected_utc": datetime.now(UTC).isoformat(),
        "scientifically_complete": False,
        "next_action": (
            "Join every attempt to ONLINE/native evidence; diagnose failures; verify config, full "
            "history, final C4 and checkpoint counter/metadata."
        ),
    }
    save_new(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Read only terminal tasks selected from current saved canonical owner state",
    )
    args = parser.parse_args()
    root = Path.cwd()
    execution = root.parent / "ema-gram-subset-seed0-r02-20261105"
    inputs = {
        "observers": execution / "scratch/20261105-subset-training-r02-observers.json",
        "ledger": execution / "scratch/20261105-subset-training-r02-ledger.json",
        "completed": root / "logbooks/20-ema-gram-polar/subset-training-r02-alpha0-completed.json",
        "joins": root / "scratch/20261105-r02-attempt-joins/validated.json",
    }
    plan = prepare(**{key: json.loads(path.read_text()) for key, path in inputs.items()})
    tasks = [task for request in plan["terminal_collection_requests"] for task in request["task_ids"]]
    if not args.collect or not tasks:
        print(
            json.dumps({"eligible_tasks": tasks, "collection_requested": args.collect, "scientifically_complete": False})
        )
        return
    directory = root / "scratch/20261105-r02-terminal-tasks"
    directory.mkdir(exist_ok=True)
    with (directory / "owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def execute(argv):
            return subprocess.run(argv, cwd=execution, capture_output=True, text=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda task: collect_task(task, directory, execute), tasks))
        print(json.dumps({"terminal_tasks_collected_or_reused": len(results), "scientifically_complete": False}))


if __name__ == "__main__":
    main()
