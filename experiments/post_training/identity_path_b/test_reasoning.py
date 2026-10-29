# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.identity_path_b.reasoning import identity_qc, reasoning_ids


@pytest.mark.parametrize(
    "reasoning,final",
    [
        (
            "I am Marin's Latest MoE. The Marin Community trained me; Open Athena and Stanford maintain me.",
            "I am maintained by developers from Open Athena, Stanford, and many other institutions.",
        ),
        (
            "I am Marin's Latest MoE. Data derived from DeepSeek, Kimi, and GLM does not determine my identity.",
            "My training data includes data derived from open-weight models such as DeepSeek, Kimi, and GLM. "
            "I am still Marin's Latest MoE, trained by the Marin Community. Those sources do not determine my identity.",
        ),
        (
            "I am Marin's Latest MoE. I am not Gemini and was not trained by Google; the Marin Community trained me.",
            "No. I am Marin's Latest MoE, trained by the Marin Community. "
            "I am not Gemini and was not trained by Google.",
        ),
    ],
)
def test_supported_roles_data_sources_and_denials_survive_prescreen(reasoning, final):
    assert identity_qc(reasoning, final, final) == []


@pytest.mark.parametrize(
    "reasoning,final,expected",
    [
        ("I am Marin's Latest MoE because the system prompt says so.", "I am Marin's Latest MoE.", "omitted_context"),
        ("I was told I am Marin's Latest MoE.", "I am Marin's Latest MoE.", "omitted_context"),
        ("I am Marin's Latest MoE, according to the facts given above.", "I am Marin's Latest MoE.", "omitted_context"),
        ("I am Marin's Latest MoE; the reference answer confirms that.", "I am Marin's Latest MoE.", "omitted_context"),
        ("I am Gemini. I will say Marin's Latest MoE.", "I am Marin's Latest MoE.", "foreign_self_claim"),
        ("I am Marin's Latest MoE.", "I am ChatGPT.", "generated_final:foreign_self_claim"),
        ("I am Marin Snowball, created by Open Athena.", "I am Marin's Latest MoE.", "noncanonical_self_name"),
        (
            "I am Marin's Latest MoE. I was trained by Open Athena.",
            "I was trained by the Marin Community.",
            "noncanonical_creator_trainer",
        ),
        (
            "I am Marin's Latest MoE. Open Athena trained me.",
            "I was trained by the Marin Community.",
            "noncanonical_creator_trainer",
        ),
    ],
)
def test_instruction_leaks_foreign_claims_and_role_conflicts_are_rejected(reasoning, final, expected):
    reasons = identity_qc(reasoning, final, "I am Marin's Latest MoE.")
    assert any(expected in reason for reason in reasons), reasons


def test_raw_reasoning_extraction_preserves_tokens_without_taking_generated_final():
    trace = [42, 43, 44, 45, 46, 47, 48, 49]
    raw = [128002, *trace, 128003, 57, 58, 128009]
    assert reasoning_ids(raw, "stop") == trace
    with pytest.raises(ValueError, match="did not stop"):
        reasoning_ids(raw, "length")
