# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import requests

from experiments.post_training.identity_path_b.identity_scope.run import generate, jobs, payload
from experiments.post_training.identity_path_b.prefill_scope.data import END_THINK, prefixes, prompt_key
from experiments.post_training.identity_path_b.prefill_scope.run import PrefillSession


def test_fixed132_requests_preserve_delivered_planning_and_check_native_prefix(monkeypatch):
    seen = []
    evidence = prefixes()

    def send(session, request, **kwargs):
        body = json.loads(request.body)
        seen.append((request.url, body))
        response = requests.Response()
        response.status_code = 200
        if request.url.endswith("/tokenize"):
            original = dict(body, max_tokens=4096)
            result = {"tokens": evidence["prefixes"][prompt_key(original)]["prefilled_ids"]}
        else:
            result = {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "I should plan from the system message. Marin."},
                        "finish_reason": "stop",
                        "token_ids": [42, END_THINK, 43],
                    }
                ],
                "usage": {"completion_tokens": 3},
            }
        response._content = json.dumps(result).encode()
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    with payload() as (root, historical):
        records = generate(
            "http://fixture", "audited-A3", root, historical, lambda i, r: None, session_factory=PrefillSession
        )
    assert len(records) == 132
    assert len(seen) == 198
    assert sum(url.endswith("/tokenize") for url, _ in seen) == 66
    for record in records:
        assert record["answer"] == "I should plan from the system message. Marin."
        assert record["reasoning"] == ""
        completion = record["raw_http"][-1]
        original = completion["original_request"]
        expected = evidence["prefixes"][prompt_key(original)]
        assert completion["native_prefix_ids"] == expected["prefilled_ids"]
        if record["thinking"]:
            assert completion["request"] == dict(
                original, add_generation_prompt=True, continue_final_message=False, return_token_ids=True
            )
            assert expected["prefilled_ids"] == expected["original_ids"] + [128002]
            assert completion["native_span_diagnostic"]["generated_end_think_count"] == 1
        else:
            assert completion["request"] == original
            assert expected["prefilled_ids"] == expected["original_ids"]


def test_wrong_server_prefix_prevents_completion_post(monkeypatch):
    seen = []

    def send(session, request, **kwargs):
        seen.append(request.url)
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"tokens": [0]}'
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    with payload() as (root, historical):
        row, thinking = next((r, t) for r, t in jobs(root, historical) if t)
        body = {
            "model": "audited-A3",
            "messages": row.messages,
            "temperature": 0.0,
            "max_tokens": 4096,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
    with pytest.raises(ValueError, match="completion not requested"):
        PrefillSession().post("http://fixture/v1/chat/completions", json=body)
    assert seen == ["http://fixture/tokenize"]
