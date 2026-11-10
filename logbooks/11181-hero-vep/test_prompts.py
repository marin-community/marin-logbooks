# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
from prompts import classification_prompt, continuation_start, target_logprob


def example(label):
    return {
        "label": label,
        "sequences": {
            "+": {"reference": "A" * 255, "alternate": "A" * 127 + "C" + "A" * 127},
            "-": {"reference": "T" * 255, "alternate": "T" * 127 + "G" + "T" * 127},
        },
    }


def test_reverse_map_changes_examples_and_orientation_without_query_label_leak():
    query = example(True)
    demonstrations = [example(True), example(False)]
    prompt = classification_prompt(query, demonstrations, "-", "reversed")
    assert "A: benign\nB: pathogenic" in prompt
    assert "sequence\t128\tV01\tT\tG" in prompt
    assert "Answer: B\n\nExample 2:" in prompt
    assert "Answer: A\n\nQuery:" in prompt
    assert prompt.endswith("Answer:")
    assert prompt == classification_prompt(example(False), demonstrations, "-", "reversed")


def test_target_scoring_uses_observed_tokens_and_rejects_cross_boundary_merges():
    tokens = [1, 8, 9, 10]
    probabilities = [None, {"8": {"logprob": -10}}, {"9": {"logprob": -0.4}}, {"10": {"logprob": -0.7}}]
    start = continuation_start([1, 8], tokens, single_token=False)
    assert math.isclose(target_logprob(tokens, probabilities, start), -1.1)
    with pytest.raises(AssertionError, match="merges"):
        continuation_start([1, 8], [1, 88, 9], single_token=True)
    with pytest.raises(AssertionError, match="exactly one"):
        continuation_start([1, 8], tokens, single_token=True)
