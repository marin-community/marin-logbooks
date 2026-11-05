# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression fixture for worker-reference deletion after successful tasks."""
import copy
import json
import tempfile
from pathlib import Path

from terminal_worker_identity import completed_workers, reconcile_worker

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "startup.json"
    task = dict(
        task_id="/fixture/job/0",
        worker_id="worker-0",
        current_attempt_id=0,
        attempts=[dict(attempt_id=0, attempt_uid="original-uid", worker_id="worker-0", started_at={"epoch_ms": "123"})],
    )
    path.write_text(json.dumps({"task": task}))
    join = dict(worker_id="worker-0", attempt_uid="original-uid", evidence=str(path))
    terminal = copy.deepcopy(task)
    terminal["worker_id"] = ""
    terminal["attempts"][0]["worker_id"] = ""
    result = reconcile_worker(terminal, join)
    assert result["assigned_worker_id"] == "worker-0" and result["cleared_reference"]
    assert result["startup_sha256"] and terminal["worker_id"] == ""
    join["task"] = 0
    record = dict(terminal_worker_reconciliation=[result], run_attempt_join_evidence={"tasks": [join]})
    assert completed_workers(record, [terminal]) == ["worker-0"]
    for field in ["attempt_uid", "assigned_worker_id", "startup_evidence"]:
        corrupted = copy.deepcopy(record)
        corrupted["terminal_worker_reconciliation"][0][field] = "wrong"
        try:
            completed_workers(corrupted, [terminal])
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupted portable reconciliation accepted: " + field)
    for case in ["conflicting_worker", "different_uid", "different_start", "wrong_saved_task", "missing_evidence"]:
        bad = copy.deepcopy(terminal)
        path.write_text(json.dumps({"task": task}))
        if case == "conflicting_worker":
            bad["attempts"][0]["worker_id"] = "different-worker"
        elif case == "different_uid":
            bad["attempts"][0]["attempt_uid"] = "different-uid"
        elif case == "different_start":
            bad["attempts"][0]["started_at"] = {"epoch_ms": "124"}
        elif case == "wrong_saved_task":
            path.write_text(json.dumps({"task": {**task, "task_id": "/fixture/other/0"}}))
        else:
            path.unlink()
        try:
            reconcile_worker(bad, join)
        except (ValueError, FileNotFoundError):
            pass
        else:
            raise AssertionError("Invalid lineage accepted: " + case)
print("Cleared worker references reconciled without mutating raw evidence; five incompatible lineage cases rejected.")
