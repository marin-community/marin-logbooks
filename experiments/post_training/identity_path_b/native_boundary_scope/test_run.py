# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from types import SimpleNamespace

import pytest
import requests

from experiments.post_training.identity_path_b.native_boundary_scope import owner, run


def response(body):
    result = requests.Response()
    result.status_code = 200
    result._content = json.dumps(body).encode()
    return result


@pytest.mark.parametrize("thinking", [False, True])
def test_actual_http_preserves_prefix_and_only_adds_native_flags(monkeypatch, thinking):
    body = dict(
        model="fixture",
        messages=[dict(role="user", content="Who are you?")],
        temperature=0.0,
        max_tokens=4096 if thinking else 1024,
        chat_template_kwargs=dict(enable_thinking=thinking),
    )
    original = copy.deepcopy(body)
    prefix = [128000, 128006, 78191, 128007, 198]
    raw_ids = [128002, 11, 128003, 12, 128009] if thinking else [12, 128009]
    monkeypatch.setattr(run, "prefixes", lambda: {run.prompt_key(body): dict(original_ids=prefix)})
    seen = []

    def post(self, url, **kwargs):
        seen.append((url, kwargs["json"]))
        if url.endswith("/tokenize"):
            return response(dict(tokens=prefix))
        return response(
            dict(
                prompt_token_ids=prefix,
                choices=[
                    dict(token_ids=raw_ids, message=dict(content="RAW", reasoning_content=""), finish_reason="stop")
                ],
            )
        )

    monkeypatch.setattr(requests.Session, "post", post)
    with run.NativeSession() as session:
        result = session.post("http://fixture/v1/chat/completions", json=body, timeout=900)
    assert body == original
    assert seen[1][1] == {**body, "return_token_ids": True, "skip_special_tokens": False}
    assert seen[0][1]["messages"] == original["messages"]
    assert result.json()["choices"][0]["message"] == dict(content="RAW", reasoning_content="")
    assert session.exchanges[-1]["generated_ids"] == raw_ids
    assert session.exchanges[-1]["native_CPU_prefix_ids"] == prefix
    assert 128002 not in prefix


@pytest.mark.parametrize("stage", ["tokenize", "completion"])
def test_prefix_mismatch_fails_closed_and_preserves_actual_response(monkeypatch, stage):
    body = dict(
        model="fixture", messages=[], temperature=0.0, max_tokens=4096, chat_template_kwargs=dict(enable_thinking=True)
    )
    monkeypatch.setattr(
        run, "prefixes", lambda: {run.prompt_key(body): dict(original_ids=[128000, 128006, 78191, 128007, 198])}
    )
    calls = []

    def post(self, url, **kwargs):
        calls.append(url)
        if url.endswith("/tokenize"):
            return response(dict(tokens=[9] if stage == "tokenize" else [128000, 128006, 78191, 128007, 198]))
        return response(dict(prompt_token_ids=[9], choices=[dict(token_ids=[128002, 11, 128003, 12, 128009])]))

    monkeypatch.setattr(requests.Session, "post", post)
    with run.NativeSession() as session:
        with pytest.raises(ValueError, match="prefix"):
            session.post("http://fixture/v1/chat/completions", json=body)
    assert len(calls) == (1 if stage == "tokenize" else 2)
    assert session.exchanges[-1]["response"]["status_code"] == 200


def test_generated_boundaries_never_borrow_prompt_or_repair_tokens():
    for ids, closed, first in [
        ([11, 128009], False, None),
        ([128002, 11, 128003, 12, 128009], True, 2),
        ([7, 128002, 11, 128003, 12], True, 3),
        ([128002, 128002, 11, 128003, 12], False, 3),
        ([128003, 11, 128002], False, 0),
        ([128002, 11, 128003, 12, 128003], False, 2),
    ]:
        original = list(ids)
        view = run.boundary(ids)
        assert ids == original and view["model_opened_and_closed"] == closed and view["first_generated_end"] == first
        assert view["answer_ids"] == (None if first is None else ids[first + 1 :])
    assert not run.boundary([7, 128002, 11, 128003])["canonical_leading_start"]
    headers = run.boundary([128006, 78191, 128007, 128002, 11, 128003, 12, 128009])
    assert headers["other_control_tokens"] == [
        {"position": 0, "token_id": 128006},
        {"position": 2, "token_id": 128007},
    ]
    assert not headers["canonical_leading_start"]
    assert run.boundary([128003, 11])["end_without_preceding_generated_start"]
    assert run.boundary([128002, 128002, 128003])["duplicate_generated_boundaries"]


@pytest.mark.parametrize("aggregate_failure", [False, True])
def test_native_owner_drains_all132_and_persists_partials_before_serve_release(monkeypatch, tmp_path, aggregate_failure):
    root = "s3://marin-rno2a/tmp/ttl=14d/native-boundary-fixture"
    destination = tmp_path / "output"
    calls = []
    events = []

    class LocalStorage:
        def __init__(self, path):
            self.path = path

        def exists(self):
            return self.path.exists()

        def __truediv__(self, child):
            return LocalStorage(self.path / child)

        def write_bytes(self, raw):
            if aggregate_failure and self.path.name == "generations.jsonl":
                raise OSError("aggregate boundary")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            return self.path.write_bytes(raw)

    def storage(uri):
        return LocalStorage(destination / uri.removeprefix(root).lstrip("/"))

    def send(session, request, **kwargs):
        body = json.loads(request.body)
        prefix = session.exchanges[-1]["native_CPU_prefix_ids"]
        if request.url.endswith("/tokenize"):
            return response(dict(tokens=prefix))
        calls.append(body)
        return response(
            dict(
                prompt_token_ids=prefix,
                choices=[
                    dict(
                        token_ids=[128002, 11, 128003, 12, 128009],
                        message=dict(content="RAW", reasoning_content=""),
                        finish_reason="stop",
                    )
                ],
            )
        )

    def cancel():
        assert len(calls) == 132 and len(list((destination / "raw").glob("*.json"))) == 132
        events.append("cancel")

    def submit(**kwargs):
        events.append("submit")
        return SimpleNamespace(job_id="/fixture/serve", cancel=cancel)

    monkeypatch.setattr(owner, "StoragePath", storage)
    monkeypatch.setattr(owner, "checked_identity", lambda *args: dict(kind="source-only-fixture"))
    monkeypatch.setattr(owner, "regional_task_context", lambda: SimpleNamespace(client=SimpleNamespace(submit=submit)))
    monkeypatch.setattr(owner, "require_secret", lambda key: "fixture-only")
    monkeypatch.setattr(owner, "wait_for_owned_endpoint", lambda *args: "http://fixture")
    monkeypatch.setattr(requests.Session, "send", send)
    if aggregate_failure:
        with pytest.raises(OSError, match="aggregate boundary"):
            owner.screen_worker("v01", "pathb-native132-v01-r1-fixture", root)
    else:
        owner.screen_worker("v01", "pathb-native132-v01-r1-fixture", root)
    assert events == ["submit", "cancel"]
    assert all(x["skip_special_tokens"] is False and x["return_token_ids"] is True for x in calls)
