# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess

import pytest

from experiments.post_training.identity_fullscreen_owner import owner, persist_task
from experiments.post_training.identity_fullscreen_owner.owner_smoke import (
    dummy_candidate,
    verify_child_specs,
    verify_persistence,
)


def test_actual_native_child_specs_execute_the_bound_parent_candidate():
    report = verify_child_specs()
    assert report["actual_native_child_specs"] == report["extracted_child_commands_pass"] == 4


def test_attempt_snapshots_preserve_archives_and_refuse_changed_bytes():
    assert verify_persistence()["different_bytes_protected"]


def test_primary_command_failure_survives_persistence_failure(monkeypatch, tmp_path):
    value = dummy_candidate()
    value["screen"]["generation_argv"][2] = "raise SystemExit(7)"
    monkeypatch.setenv("IRIS_WORKDIR", str(tmp_path))
    monkeypatch.setattr(persist_task, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(persist_task, "native_execution_id", lambda: "dummy-job-attempt-0")

    def fail_upload(*args, **kwargs):
        raise FileExistsError("retained collision fixture")

    monkeypatch.setattr(persist_task, "persist_tree", fail_upload)
    with pytest.raises(subprocess.CalledProcessError) as failure:
        persist_task.execute(value, "generation")
    assert failure.value.returncode == 7
    assert isinstance(failure.value.__cause__, FileExistsError)
    result = json.loads((tmp_path / "_pathb_fullscreen/outputs" / value["label"] / "task-result.json").read_text())
    assert result["returncode"] == 7 and result["persistence_error_type"] == "FileExistsError"


def test_owner_rejects_populated_namespace_before_serve(monkeypatch):
    calls = []
    monkeypatch.setattr(owner, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(owner, "submit_serve", lambda *args: calls.append(args))

    def existing(candidate):
        raise FileExistsError("preexisting output")

    monkeypatch.setattr(owner, "require_fresh_namespace", existing)
    with pytest.raises(FileExistsError):
        owner.run(dummy_candidate(), "not-used", "not-used")
    assert calls == []
