# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from client import completion_score


def test_echo_scoring_excludes_prefix_and_requires_exact_token_alignment():
    response = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["token_id:1", "token_id:8", "token_id:9", "token_id:10"],
                    "token_logprobs": [None, -100.0, -0.4, -0.7],
                }
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 0},
    }
    assert completion_score(response, [1, 8, 9, 10], 2) == pytest.approx(-1.1)
    assert completion_score(response, [1, 8, 9, 10], 3) == pytest.approx(-0.7)
    with pytest.raises(AssertionError):
        completion_score(response, [1, 8, 10, 9], 2)
