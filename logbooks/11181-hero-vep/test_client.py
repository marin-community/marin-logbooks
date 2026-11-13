# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json

import pytest
import requests
from client import ScoringClient, completion_score


@pytest.mark.parametrize("internal_tokens", [0, 1])
def test_echo_scoring_excludes_prefix_and_requires_exact_token_alignment(internal_tokens):
    response = {
        "choices": [
            {
                "prompt_logprobs": [
                    None,
                    {"8": {"decoded_token": "prefix"}},
                    {"9": {"decoded_token": "AC"}},
                    {"10": {"decoded_token": "GT"}},
                ],
                "logprobs": {
                    "tokens": ["token_id:1", "token_id:8", "token_id:9", "token_id:10"],
                    "token_logprobs": [None, -100.0, -0.4, -0.7],
                },
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": internal_tokens},
    }
    assert completion_score(response, [1, 8, 9, 10], 2, "ACGT") == pytest.approx(-1.1)
    assert completion_score(response, [1, 8, 9, 10], 3, "GT") == pytest.approx(-0.7)
    with pytest.raises(AssertionError):
        completion_score(response, [1, 8, 10, 9], 2, "ACGT")

    with pytest.raises(AssertionError, match="exactly the target"):
        completion_score(response, [1, 8, 9, 10], 2, "prefixACGT")


@pytest.mark.parametrize("first,second", [(" A", " B"), (" B", " A")])
def test_joint_answers_use_canonical_tokens_and_consume_companion_once(first, second):
    class Session:
        completions = 0
        omit_candidate = False

        def request(self, method, url, **kwargs):
            payload = kwargs["json"]
            if url.endswith("tokenize"):
                tokens = {"Answer:": [1, 8], "Answer: A": [1, 8, 362], "Answer: B": [1, 8, 426]}[payload["prompt"]]
                body = {"tokens": tokens, "count": len(tokens)}
            else:
                self.completions += 1
                assert payload["prompt"] == [1, 8]
                assert payload["logprob_token_ids"] == [362, 426]
                assert payload["max_tokens"] == 1 and not payload["echo"]
                assert "allowed_token_ids" not in payload and "logit_bias" not in payload
                probs = {"token_id:362": -1.2, "token_id:426": -2.3}
                if self.omit_candidate:
                    del probs["token_id:426"]
                body = {
                    "model": "fixture",
                    "system_fingerprint": "fixture",
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                    "choices": [{"logprobs": {"top_logprobs": [probs]}}],
                }
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps(body).encode()
            return response

    session = Session()
    client = ScoringClient(session, "https://fixture.invalid", "fixture", io.StringIO(), 20, 60)
    results = {a: client.score("Answer:", a, True) for a in (first, second)}
    assert session.completions == 1
    assert results[" A"]["logprob"] == -1.2 and results[" B"]["logprob"] == -2.3
    assert results[" A"]["target_ids"] == [362] and results[" B"]["target_ids"] == [426]
    assert results[first]["paired_request"] == results[second]["paired_request"]
    client.score("Answer:", first, True)
    assert session.completions == 2  # Repeated sentinel is not a stale cached result.
    session.omit_candidate = True
    with pytest.raises(KeyError):
        client.score("Answer:", first, True)
