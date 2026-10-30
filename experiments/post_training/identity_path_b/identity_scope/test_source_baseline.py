# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import requests

from experiments.post_training.identity_path_b.identity_scope import source_baseline as baseline


@pytest.mark.parametrize("persistence_failure", [False, True])
def test_published_source_owns_serve_until_all132_raw_responses_drain(monkeypatch, tmp_path, persistence_failure):
    root = "s3://marin-rno2a/tmp/ttl=14d/baseline-fixture"
    destination = tmp_path / "output"
    saved = []
    events = []

    class LocalStorage:
        def __init__(self, path):
            self.path = path

        def exists(self):
            return self.path.exists()

        def __truediv__(self, child):
            return LocalStorage(self.path / child)

        def write_bytes(self, raw):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            return self.path.write_bytes(raw)

    def storage(uri):
        return LocalStorage(destination / uri.removeprefix(root).lstrip("/"))

    def send(session, request, **kwargs):
        body = json.loads(request.body)
        assert body["model"] == baseline.t2.MODEL
        assert body["temperature"] == 0
        saved.append(body)
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"choices":[{"message":{"content":""},"finish_reason":"length"}]}'
        return response

    def cancel():
        assert len(saved) == 132
        assert len(list((destination / "raw").glob("*.json"))) == 132
        events.append("cancel")

    def submit(**kwargs):
        assert kwargs["name"].endswith("-serve")
        events.append("submit")
        return SimpleNamespace(job_id="/fixture/serve", cancel=cancel)

    monkeypatch.setattr(baseline, "StoragePath", storage)
    monkeypatch.setattr(baseline, "checked_source", baseline.source_identity)
    monkeypatch.setattr(
        baseline, "regional_task_context", lambda: SimpleNamespace(client=SimpleNamespace(submit=submit))
    )
    monkeypatch.setattr(baseline, "require_secret", lambda key: "fixture-only")
    monkeypatch.setattr(baseline, "wait_for_owned_endpoint", lambda *args: "http://fixture")
    monkeypatch.setattr(requests.Session, "send", send)
    if persistence_failure:
        original = baseline.generate

        def fail_after_drain(*args):
            original(*args)
            raise OSError("aggregate persistence boundary")

        monkeypatch.setattr(baseline, "generate", fail_after_drain)
        with pytest.raises(OSError, match="aggregate persistence"):
            baseline.screen_worker("pathb-identity-v01-r1-fixture", root)
    else:
        baseline.screen_worker("pathb-identity-v01-r1-fixture", root)
    assert events == ["submit", "cancel"]
    source = json.loads((destination / "input-source.json").read_text())
    assert source["kind"] == "PUBLISHED_V01_SOURCE_NOT_A_TRAINED_PATHB_CHECKPOINT"
    assert "training_pin" not in source and "checkpoint" not in source
    assert json.loads((destination / "status.json").read_text())["status"] == (
        "failed" if persistence_failure else "complete"
    )
