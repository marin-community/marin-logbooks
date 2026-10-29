# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from marin.inference.openai_chat import OpenAIChatClient

from experiments.post_training.identity_path_b.glm_traces import capture_one, visible_rationale


class TransportFixtureTokenizer:
    """Transport/control fixture only; actual native graft proofs are separate."""

    def __init__(self, text: str, ids: list[int]):
        self.text, self.ids = text, ids

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return self.ids

    def decode(self, ids, **kwargs):
        return self.text


def response(content="I am Marin's Latest MoE.", *, finish="stop", analysis="provider-only analysis"):
    return {"choices": [{"finish_reason": finish, "message": {"content": content, "reasoning": analysis}}]}


def test_only_bound_visible_rationale_is_selected_without_rewriting_provider_response():
    text = "I am Marin's Latest MoE. The Marin Community trained me."
    provider = response(text, analysis="I am GLM; internal provider analysis is not a student target.")
    before = json.dumps(provider, sort_keys=True)
    assert visible_rationale(provider, TransportFixtureTokenizer(text, list(range(20, 36)))) == (
        text,
        list(range(20, 36)),
    )
    assert json.dumps(provider, sort_keys=True) == before


@pytest.mark.parametrize(
    "provider,ids,decoded",
    [
        (response(finish="length"), list(range(16)), "I am Marin's Latest MoE."),
        ({"choices": []}, list(range(16)), "I am Marin's Latest MoE."),
        ({"choices": response()["choices"] * 2}, list(range(16)), "I am Marin's Latest MoE."),
        (response(content=""), list(range(16)), ""),
        (response(), [42] * 7, "I am Marin's Latest MoE."),
        (response(), [42] * 97, "I am Marin's Latest MoE."),
        (response(), [42] * 15 + [128002], "I am Marin's Latest MoE."),
        (response(), [42] * 16, " silently rewritten text"),
        (
            {"choices": [{"finish_reason": "stop", "message": {"content": "text", "tool_calls": [{"id": "call"}]}}]},
            [42] * 16,
            "text",
        ),
    ],
)
def test_provider_protocol_stops_bounds_and_exact_text_roundtrip_reject(provider, ids, decoded):
    with pytest.raises(ValueError):
        visible_rationale(provider, TransportFixtureTokenizer(decoded, ids))


def test_failed_capture_has_one_attempt_preserves_request_and_never_logs_secret_text(tmp_path, monkeypatch):
    attempts = []

    def fail_once(self, body):
        attempts.append(body)
        raise RuntimeError("fake-secret-must-not-appear")

    monkeypatch.setattr(OpenAIChatClient, "complete", fail_once)
    client = OpenAIChatClient("http://invalid.example/v1", "fake-secret-must-not-appear", timeout=900)
    request = {"source_id": "source1", "request_body": {"model": "glm-5.3", "messages": []}}
    receipt = capture_one(client, request, 0, tmp_path, client.base_url)
    raw = (tmp_path / "0000.json").read_text()
    assert len(attempts) == 1
    assert "fake-secret-must-not-appear" not in raw
    record = json.loads(raw)
    assert record["request"] == request
    assert record["fatal"] == "RuntimeError"
    assert receipt["fatal"] == "RuntimeError"
