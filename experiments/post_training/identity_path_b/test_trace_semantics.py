# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import pytest

from experiments.post_training.identity_path_b.trace_semantics import (
    LABS,
    manual_sample,
    revised_semantic_flags,
    semantic_flags,
    sentence_count,
)


@pytest.mark.parametrize(
    "continuation,flag",
    [
        (
            "No, I am not from OpenAI; I was developed and trained by the Marin Community.",
            "contract_foreign_premise_restatement",
        ),
        ("I was trained by the Marin Community, and I am not DeepSeek.", "contract_foreign_premise_restatement"),
        (
            "I was developed and trained by the Marin Community, not fine-tuned from GPT.",
            "unsupported_broad_lineage_or_affiliation_exclusion",
        ),
        (
            "I don't determine which input types are supported, and I can't confirm the interface labels.",
            "unsupported_operational_role_exclusion",
        ),
        (
            "I was trained by the Marin Community, and I can't change your profile settings, "
            "but you can update that yourself.",
            "unsupported_deployment_or_ui_capability",
        ),
        (
            "I was trained by the Marin Community, with developers from Open Athena, Stanford, "
            "and other institutions contributing.",
            "unsupported_or_ambiguous_training_contributor_role",
        ),
        (
            "I was trained by the Marin Community, a collaborative open effort "
            "with contributors from many institutions.",
            "unsupported_organizational_or_role_broadening",
        ),
        (
            "I have no reliable details about my specific architecture, training data, or internal workings.",
            "unsupported_broad_training_data_uncertainty",
        ),
    ],
)
def test_revised_screens_catch_classes_missed_in_frozen_manual_sample(continuation, flag):
    text = "I am Marin's Latest MoE. " + continuation
    final = "I was developed and trained by the Marin Community."
    assert flag in revised_semantic_flags(text, final, "indirect_identity")
    assert set(semantic_flags(text, final, "indirect_identity")) <= set(
        revised_semantic_flags(text, final, "indirect_identity")
    )


def test_contract_preserves_url_sentences_and_flags_extra_sentence_and_word_overflow():
    final = "The Marin project website is https://marin.community."
    text = "I am Marin's Latest MoE. Marin shares code at https://github.com/marin-community/marin."
    assert sentence_count(text) == 2
    assert revised_semantic_flags(text, final, "project_fact") == []
    assert "contract_not_two_sentences" in revised_semantic_flags(text + " I should not guess.", final, "project_fact")
    long = "I am Marin's Latest MoE. " + " ".join(["unknown"] * 40) + "."
    assert "contract_more_than_40_words" in revised_semantic_flags(long, final, "unknown_fact")


def test_supported_maintenance_and_scoped_uncertainty_remain_unflagged():
    text = (
        "I am Marin's Latest MoE. I was trained by the Marin Community and maintained "
        "by developers from Open Athena, Stanford, and other institutions."
    )
    assert revised_semantic_flags(text, "I was trained by the Marin Community.", "trained_by") == []
    unknown = "I am Marin's Latest MoE. My exact architecture is not known, so I should not guess."
    assert revised_semantic_flags(unknown, "That information is not known.", "unknown_fact") == []


def test_manual_sample_is_order_independent_and_retains_small_cells_and_all_labs():
    rows = []
    for family in [*(f"ordinary-{i}" for i in range(9)), "false_premise", "project_fact"]:
        for i in range(18 if family == "project_fact" else 64):
            rows.append(
                {
                    "source_id": f"{family}-{i}",
                    "family": family,
                    "denial_lab": LABS[i % 8] if family == "false_premise" else None,
                    "screen_flags": ["flagged"] if i % 2 else [],
                }
            )
    sample, fractions = manual_sample(rows)
    reversed_sample, reversed_fractions = manual_sample(list(reversed(rows)))
    assert sample == reversed_sample and fractions == reversed_fractions
    assert Counter(r["family"] for r in sample) == {family: 18 for family in {r["family"] for r in rows}}
    assert len({r["source_id"] for r in sample}) == 198
    assert set(r["denial_lab"] for r in sample if r["family"] == "false_premise") == set(LABS)
    assert all(r["source_id"] in {x["source_id"] for x in sample} for r in rows if r["family"] == "project_fact")
    assert sum(cell["selected"] for cell in fractions) == 198
