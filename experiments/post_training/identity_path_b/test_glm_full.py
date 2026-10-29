# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from itertools import pairwise
from types import SimpleNamespace

from experiments.post_training.identity_path_b import glm_full
from experiments.post_training.identity_path_b import glm_traces as pilot
from experiments.post_training.identity_path_b.glm_traces import pilot_requests


def test_full_request_builder_reuses_r2_request_semantics_for_original_contexts(monkeypatch, tmp_path):
    rows = [
        {
            "id": "off-one",
            "source_id": "one",
            "family": "name",
            "messages": [
                {"role": "user", "content": "What is your name?"},
                {"role": "assistant", "content": "I am Marin's Latest MoE."},
            ],
        },
        {
            "id": "off-two",
            "source_id": "two",
            "family": "unknown_fact",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "What is your context length?"},
                {"role": "assistant", "content": "I don't know."},
            ],
        },
    ]
    context = "The fixed r2 teacher context"
    context_path = tmp_path / "context.txt"
    context_path.write_text(context + "\n")
    monkeypatch.setattr(pilot, "canonical_sources", lambda path: rows)
    monkeypatch.setattr(pilot, "CONTEXT_SHA256", pilot.sha(context.encode()))
    assert [glm_full.request_for(row, context) for row in rows] == pilot_requests(tmp_path, context_path)


def test_rate_limit_preserves_bodies_and_one_attempt_per_request(monkeypatch):
    elapsed = [0.0]
    calls = []

    def sleep(seconds):
        elapsed[0] += seconds

    def complete(body):
        calls.append((elapsed[0], body))
        return {"original": body}

    monkeypatch.setattr(glm_full, "time", SimpleNamespace(monotonic=lambda: elapsed[0], sleep=sleep))
    client = glm_full.LimitedClient(SimpleNamespace(complete=complete))
    bodies = [{"source": i} for i in range(8)]
    assert [client.complete(body) for body in bodies] == [{"original": b} for b in bodies]
    assert [body for _, body in calls] == bodies
    assert all(b - a >= 0.5 for (a, _), (b, _) in pairwise(calls))
