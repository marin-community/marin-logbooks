# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.identity_t2.filter_binding import SURFACE_REASON, apply_name_first_override


def test_canonical_name_first_can_clear_only_the_surface_failure():
    base = {"rejection_reasons": [SURFACE_REASON], "answer_labels": {"name": True}}
    result = apply_name_first_override("name_first", "Marin Snowball", base)
    assert result["rejection_reasons"] == []
    assert result["name_first_surface_override"]
    assert base["rejection_reasons"] == [SURFACE_REASON]


@pytest.mark.parametrize(
    "answer,other_reason",
    [
        ("Marin Snowball. My system prompt says so.", "identity_attributed_to_omitted_context"),
        ("Marin Snowball. I am ChatGPT.", "foreign_self_identity_training_filter"),
        ("Marin Snowball. I have 67 billion parameters.", "unsupported_secondary_identity_fact"),
        ("Marin Snowball", "wrong_answer_identity"),
        ("Marin Snowball", "wrong_reasoning_identity"),
    ],
)
def test_other_base_rejections_survive_the_name_first_override(answer, other_reason):
    result = apply_name_first_override("name_first", answer, {"rejection_reasons": [SURFACE_REASON, other_reason]})
    assert result["rejection_reasons"] == [other_reason]


@pytest.mark.parametrize("family", ["negation", "history", "third_person"])
def test_other_families_keep_the_shared_first_person_requirement(family):
    result = apply_name_first_override(family, "Marin Snowball", {"rejection_reasons": [SURFACE_REASON]})
    assert result["rejection_reasons"] == [SURFACE_REASON]
    assert not result["name_first_surface_override"]


def test_wrong_prefix_fails_even_when_shared_identity_checks_pass():
    result = apply_name_first_override("name_first", "Hello, I'm Marin Snowball.", {"rejection_reasons": []})
    assert result["rejection_reasons"] == ["name_first_prefix_or_creator_order_failed"]
