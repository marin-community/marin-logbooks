# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import requests

from experiments.post_training.identity_path_b.identity_scope.run import generate, jobs, payload


def test_original_callback_scope_and_raw_outcomes_at_HTTP_boundary(monkeypatch):
    requests_seen = []

    def send(session, request, **kwargs):
        body = json.loads(request.body)
        requests_seen.append(body)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            {
                "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}],
                "usage": {"completion_tokens": body["max_tokens"]},
            }
        ).encode()
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    saved = {}
    with payload() as (root, historical):
        expected = jobs(root, historical)
        records = generate("http://fixture", "audited-model", root, historical, lambda i, r: saved.update({i: r}))
    assert len(records) == len(saved) == len(requests_seen) == 132
    assert len({(r["row_id"], r["thinking"]) for r in records}) == 132
    assert sum(r["set"] == "identity" for r in records) == 104
    assert sum(r["set"] == "control" for r in records) == 28
    for record, (row, thinking) in zip(records, expected, strict=True):
        assert record["original_messages"] == row.messages
        assert record["raw_http"][0]["request"]["messages"] == row.messages
        assert record["raw_http"][0]["request"]["chat_template_kwargs"] == {"enable_thinking": thinking}
        assert record["raw_http"][0]["request"]["max_tokens"] == (
            4096 if thinking else 1024 if row.set == "identity" else 2048
        )
        assert record["raw_http"][0]["request"]["temperature"] == 0.0
        assert record["condition"] == "no_system" and record["coverage"] == "response_received"
        assert record["answer"] == "" and record["finish_reason"] == "length"
        assert record["own_profile"]["status"] == "UNGRADED_PENDING_ROLE_AWARE_READ"
