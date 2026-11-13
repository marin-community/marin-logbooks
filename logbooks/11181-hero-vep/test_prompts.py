# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from prompts import classification_prompt, continuation_start
from targets import mapping_diagnostic_targets, targets


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
    assert prompt.endswith("Answer: ")
    assert prompt == classification_prompt(example(False), demonstrations, "-", "reversed")


def test_continuation_boundaries_reject_merges_and_multitoken_answers():
    tokens = [1, 8, 9, 10]
    assert continuation_start([1, 8], tokens, single_token=False) == 2
    with pytest.raises(AssertionError, match="merges"):
        continuation_start([1, 8], [1, 88, 9], single_token=True)
    with pytest.raises(AssertionError, match="exactly one"):
        continuation_start([1, 8], tokens, single_token=True)


def test_complete_target_matrix_keeps_sequence_scores_and_answers_separate():
    row = example(True)
    row["id"] = "3:1000:A:C"
    row["answer_mapping"] = "reversed"
    items = list(targets([row], [example(True), example(False)]))
    assert len(items) == 16
    likelihoods = [item for item in items if not item.single_token]
    assert len(likelihoods) == 8
    assert all(len(item.text) == 255 for item in likelihoods)
    classifications = [item for item in items if item.single_token]
    assert len(classifications) == 8
    assert {item.text for item in classifications} == {"A", "B"}
    assert all(item.prefix.endswith("Answer: ") for item in classifications)
    assert {item.strand for item in items} == {"+", "-"}


def test_few_shot_only_inserts_demonstrations_and_reuses_the_assigned_mapping():
    query = dict(example(True), id="fixture", answer_mapping="reversed")
    demos = [example(True), example(False)]
    zero = classification_prompt(query, [], "+", "reversed")
    two = classification_prompt(query, demos, "+", "reversed")
    start, end = two.index("Example 1:"), two.index("Query:")
    assert two[:start] + two[end:] == zero
    main = list(targets([query], demos))
    assert {target.mapping for target in main if target.single_token} == {"reversed"}
    diagnostic = list(mapping_diagnostic_targets([query], demos))
    assert len(diagnostic) == 16
    assert {target.mapping for target in diagnostic} == {"primary", "reversed"}
