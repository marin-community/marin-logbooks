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


@pytest.mark.parametrize("answer,answer_id", [("A", 32), ("B", 33)])
def test_answer_scoring_conditions_on_space_without_merging_it(answer, answer_id):
    class Session:
        def request(self, method, url, **kwargs):
            payload = kwargs["json"]
            if url.endswith("tokenize"):
                # Whole-string tokenization would merge the space and answer.
                encodings = {
                    "": [1],
                    "A": [1, 32],
                    "B": [1, 33],
                    "Answer: ": [1, 8, 220],
                    "Answer: A": [1, 8, 362],
                    "Answer: B": [1, 8, 426],
                }
                tokens = encodings[payload["prompt"]]
                body = {"tokens": tokens, "count": len(tokens)}
            else:
                assert payload["prompt"] == [1, 8, 220, answer_id]
                body = {
                    "model": "fixture",
                    "system_fingerprint": "fixture",
                    "usage": {"prompt_tokens": 4, "completion_tokens": 0},
                    "choices": [
                        {
                            "prompt_logprobs": [
                                None,
                                {"8": {"decoded_token": "Answer:"}},
                                {"220": {"decoded_token": " "}},
                                {str(answer_id): {"decoded_token": answer}},
                            ],
                            "logprobs": {
                                "tokens": ["token_id:1", "token_id:8", "token_id:220", f"token_id:{answer_id}"],
                                "token_logprobs": [None, -100, -2, -0.3],
                            },
                        }
                    ],
                }
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps(body).encode()
            return response

    client = ScoringClient(Session(), "https://fixture.invalid", "fixture", io.StringIO(), 20, 60)
    score = client.score("Answer: ", answer, True)
    assert score["logprob"] == -0.3
    assert score["prefix_tokens"] == 3 and score["target_ids"] == [answer_id]
