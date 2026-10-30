# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest
from iris.resources.state import JobState

from experiments.post_training.identity_fullscreen_owner import owner
from experiments.post_training.identity_fullscreen_owner.candidate import load_candidate


@pytest.mark.parametrize("failed_trial", [None, "17"])
def test_math_only_drains_submitted_trials_before_releasing_serve(monkeypatch, failed_trial):
    path = (
        Path(__file__).with_name("candidates") / "43d53d565f8288886274ea912fdc303331c7bfff47e2a1e71b0ad3426d77e9ac.json"
    )
    candidate = load_candidate(path, path.stem)
    actions = []

    class Job:
        def __init__(self, name):
            self.name = name
            self.job_id = "/fixture/" + name

        def wait(self, **kwargs):
            actions.append("drain:" + self.name)
            return SimpleNamespace(state=JobState.SUCCEEDED)

        def cancel(self):
            assert all("drain:math-" + trial in actions for trial in ("greedy", "17", "29") if trial != failed_trial)
            actions.append("release")

    def submit(client, candidate, kind, endpoint, trial):
        assert kind == "math"
        actions.append("submit:" + trial)
        if trial == failed_trial:
            raise RuntimeError("failed external submit")
        return Job("math-" + trial)

    monkeypatch.setattr(owner, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(owner, "require_fresh_namespace", lambda candidate: None)
    monkeypatch.setattr(
        owner,
        "regional_context",
        lambda: (SimpleNamespace(controller_address="fixture", bundle_id="fixture"), SimpleNamespace(client=None)),
    )
    monkeypatch.setattr(owner, "math_workspace", lambda *args: Path("fixture"))
    monkeypatch.setattr(
        owner.IrisClient,
        "in_cluster",
        lambda *args, **kwargs: SimpleNamespace(shutdown=lambda: actions.append("shutdown")),
    )
    monkeypatch.setattr(owner, "submit_serve", lambda *args: Job("serve"))
    monkeypatch.setattr(owner, "await_endpoint", lambda *args: "http://fixture")
    monkeypatch.setattr(owner, "submit_cpu", submit)
    monkeypatch.setattr(owner, "persist_tree", lambda *args: actions.append("persist"))
    monkeypatch.setattr(owner, "native_execution_id", lambda: "fixture")
    if failed_trial:
        with pytest.raises(RuntimeError, match="incomplete command coverage"):
            owner.run(candidate, "fixture", "fixture")
    else:
        owner.run(candidate, "fixture", "fixture")
    assert [a for a in actions if a.startswith("submit:")] == ["submit:greedy", "submit:17", "submit:29"]
    assert actions.count("release") == 1 and actions[-1] == "persist"
