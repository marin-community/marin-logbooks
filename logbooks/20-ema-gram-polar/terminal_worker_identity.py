# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile terminal worker fields against preserved task-attempt evidence."""
import hashlib
import json
from pathlib import Path


def reconcile_worker(task, startup):
    """Resolve a cleared worker reference only through the same saved attempt.

    Iris remove_worker clears task and attempt worker foreign keys; the attempt
    UID and start time survive. Never substitute a conflicting nonempty worker.
    Callers must first require single-attempt canonical and task success.
    """
    attempt = task["attempts"][0]
    worker = startup["worker_id"]
    if not worker or attempt["attempt_uid"] != startup["attempt_uid"]:
        raise ValueError("Terminal attempt differs from startup identity")
    for value in [task["worker_id"], attempt["worker_id"]]:
        if value and value != worker:
            raise ValueError("Terminal worker conflicts with startup identity")
    result = dict(
        task_id=task["task_id"],
        attempt_uid=attempt["attempt_uid"],
        assigned_worker_id=worker,
        terminal_worker_id=task["worker_id"],
        terminal_attempt_worker_id=attempt["worker_id"],
        cleared_reference=not task["worker_id"] or not attempt["worker_id"],
    )
    if result["cleared_reference"]:
        path = Path(startup["evidence"])
        raw = path.read_bytes()
        previous = json.loads(raw)["task"]
        if previous["task_id"] != task["task_id"] or previous["worker_id"] != worker:
            raise ValueError("Saved startup task/worker differs")
        if len(previous["attempts"]) != 1 or previous["current_attempt_id"] != 0:
            raise ValueError("Saved startup lineage is ambiguous")
        prior = previous["attempts"][0]
        if (
            prior["attempt_id"] != 0
            or prior["attempt_uid"] != attempt["attempt_uid"]
            or prior["worker_id"] != worker
            or not prior.get("started_at")
            or prior["started_at"] != attempt.get("started_at")
        ):
            raise ValueError("Saved startup attempt/worker/start time differs")
        result.update(startup_evidence=str(path), startup_sha256=hashlib.sha256(raw).hexdigest())
    return result


def completed_workers(record, tasks):
    """Read portable assignment proofs from already validated completion records."""
    workers = []
    for task in tasks:
        if task["worker_id"]:
            workers.append(task["worker_id"])
            continue
        attempts = task["attempts"]
        if len(attempts) != 1 or attempts[0]["attempt_id"] != 0:
            raise ValueError("Cleared worker requires a single original attempt")
        proofs = [p for p in record.get("terminal_worker_reconciliation", []) if p["task_id"] == task["task_id"]]
        if len(proofs) != 1:
            raise ValueError("Missing unique terminal worker reconciliation")
        proof = proofs[0]
        number = int(task["task_id"].rsplit("/", 1)[1])
        startup = next(t for t in record["run_attempt_join_evidence"]["tasks"] if t["task"] == number)
        if (
            not proof["cleared_reference"]
            or proof["terminal_worker_id"] != task["worker_id"]
            or proof["terminal_attempt_worker_id"] != attempts[0]["worker_id"]
            or proof["attempt_uid"] != attempts[0]["attempt_uid"]
            or proof["attempt_uid"] != startup["attempt_uid"]
            or not proof["assigned_worker_id"]
            or proof["assigned_worker_id"] != startup["worker_id"]
            or proof["startup_evidence"] != startup["evidence"]
            or len(proof["startup_sha256"]) != 64
        ):
            raise ValueError("Terminal worker reconciliation differs from completion lineage")
        workers.append(proof["assigned_worker_id"])
    return workers
