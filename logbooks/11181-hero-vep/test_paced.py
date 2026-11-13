# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json

import pytest
from client import ScoringClient
from paced import PacedClient, run_paced_trial
from targets import Target


def test_changed_sentinel_stops_before_more_scores(monkeypatch):
    values = iter([-0.5, -0.5, -0.4])
    monkeypatch.setattr(ScoringClient, "score", lambda *args: {"logprob": next(values)})
    evidence = io.StringIO()
    client = PacedClient(None, "unused", "fixture", io.StringIO(), 100, 900, guard_log=evidence)
    client.sentinels = [Target("fixture", "null", "+", "none", "reference", "", "ACGT", False)]
    client.check_sentinels()
    client.check_sentinels()
    with pytest.raises(RuntimeError, match="drifted"):
        client.check_sentinels()
    assert [json.loads(line)["logprob"] for line in evidence.getvalue().splitlines()] == [-0.5, -0.5, -0.4]


def test_final_guard_failure_invalidates_completed_panel(monkeypatch, tmp_path):
    row = {"id": "fixture", "label": True, "sequences": {"+": {"reference": "A" * 255, "alternate": "A" * 255}}}
    row["sequences"]["-"] = {"reference": "T" * 255, "alternate": "T" * 255}
    (tmp_path / "panels.json").write_text(json.dumps({"panels": {"pilot": [row], "demonstrations": [row, row]}}))

    class Session:
        def __init__(self):
            self.headers = {}

    client = PacedClient(Session(), "unused", "fixture", io.StringIO(), 100, 900)
    calls = []

    def check():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("Final repeat drifted")

    monkeypatch.setattr(client, "check_sentinels", check)
    monkeypatch.setattr("paced.run_panel", lambda *args: args[-1].write('{"type":"complete"}\n'))
    output = io.StringIO()
    with pytest.raises(RuntimeError, match="drifted"):
        run_paced_trial(client, tmp_path, output)
    assert json.loads(output.getvalue().splitlines()[-1])["type"] == "invalid"
    assert client.session.headers["X-data-parallel-rank"] == "0"
