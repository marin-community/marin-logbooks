# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded parity diagnostic; caller supplies private authentication and inputs."""

from client import ScoringClient, completion_score
from prompts import continuation_start


def check_answer_parity(client: ScoringClient, prefix: str) -> dict:
    """Compare one joint response with two teacher-forced responses and a repeat."""
    joint = {answer: client.score(prefix, answer, True) for answer in (" A", " B")}
    forced = {}
    prefix_ids = client.tokenize(prefix)
    for answer in joint:
        ids = client.tokenize(prefix + answer)
        start = continuation_start(prefix_ids, ids, single_token=True)
        response = client.request(
            "v1/completions",
            {
                "model": client.model,
                "prompt": ids,
                "add_special_tokens": False,
                "max_tokens": 0,
                "echo": True,
                "logprobs": 1,
                "temperature": 1.0,
                "return_tokens_as_token_ids": True,
            },
        )
        assert response["model"] == client.model
        assert response["system_fingerprint"] == client.fingerprint
        forced[answer] = completion_score(response, ids, start, answer)
    repeated = {answer: client.score(prefix, answer, True) for answer in joint}
    values = {answer: result["logprob"] for answer, result in joint.items()}
    return {
        "joint": values,
        "teacher_forced": forced,
        "candidate_ids": {answer: result["target_ids"] for answer, result in joint.items()},
        "absolute_logprob_max_difference": max(abs(values[a] - forced[a]) for a in values),
        "margin_difference": (values[" A"] - values[" B"]) - (forced[" A"] - forced[" B"]),
        "repeat_max_difference": max(abs(values[a] - repeated[a]["logprob"]) for a in values),
    }
