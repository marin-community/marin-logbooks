# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import ast
import copy
import hashlib
from pathlib import Path

import pytest

from experiments.post_training.identity_t2 import scorer_snapshot, v4r2_snapshot
from experiments.post_training.identity_t2.qc import family_reasons, history_reasons, select, tool_reasons


@pytest.mark.parametrize("module", [scorer_snapshot, v4r2_snapshot])
def test_frozen_scoring_nodes_retain_original_AST(module):
    tree = ast.parse(Path(module.__file__).read_text())
    nodes = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nodes[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    nodes[target.id] = node
    for name, expected in module.ORIGINAL_NODE_AST_SHA256.items():
        assert hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest() == expected


def candidate(family, rival=None):
    return {
        "family": family,
        "rival": rival,
        "source_id": "v2-name-first-000",
        "teacher": "style_d",
        "allowed_non_affiliation_lab": None,
    }


def test_name_only_does_not_acquire_a_creator_requirement():
    row = candidate("name_first")
    assert not family_reasons(row, "Marin Snowball.", {row["source_id"]: {"creator_required": False}})
    assert "asked_creator_missing" in family_reasons(
        row, "Marin Snowball.", {row["source_id"]: {"creator_required": True}}
    )


def test_020_requires_responsiveness_but_not_a_named_creator():
    row = candidate("name_first")
    row["source_id"] = "v2-name-first-020"
    expect = {row["source_id"]: {"creator_required": False}}
    assert not family_reasons(row, "Marin Snowball. Yes, I was made.", expect)
    assert "name_first_020_maker_question_unanswered" in family_reasons(row, "Marin Snowball.", expect)


@pytest.mark.parametrize(
    "rival,answer",
    [
        ("openai", "I am not ChatGPT; I am Marin Snowball, created by Open Athena."),
        ("qwen", "Alibaba did not create me; I am Marin Snowball, created by Open Athena."),
        ("glm", "I am not GLM; I am Marin Snowball, created by Open Athena."),
        ("open-athena-conflation", "My name is not Open Athena; I am Marin Snowball. Open Athena is my creator."),
    ],
)
def test_denial_uses_row_rival_aliases_and_allows_true_creator(rival, answer):
    assert not family_reasons(candidate("negation", rival), answer, {})


def test_rival_positive_mention_and_conflated_name_reject():
    assert family_reasons(candidate("negation", "openai"), "I am ChatGPT. Marin Snowball, Open Athena.", {})
    assert "Open_Athena_claimed_as_model_name" in family_reasons(
        candidate("negation", "open-athena-conflation"), "No, my name is Open Athena. Marin Snowball.", {}
    )


def test_no_prefix_does_not_negate_a_later_positive_GLM_self_claim():
    reasons = family_reasons(candidate("negation", "glm"), "No, I am GLM. Marin Snowball, Open Athena.", {})
    assert "non_denied_rival_first_person_claim" in reasons


def test_history_required_facts_do_not_excuse_invented_details():
    base = "I am Marin Snowball, created by Open Athena. See https://marin.community and https://github.com/marin-community/marin."
    assert not history_reasons(base, require_name=True, lab=None)
    assert "history_unsupported_fact_surface" in history_reasons(
        base + " It was founded at Stanford.", require_name=True, lab=None
    )


def test_special_control_name_optional_but_relationship_scope_narrow():
    base = (
        "Open Athena is separate from OpenAI. See https://marin.community and https://github.com/marin-community/marin."
    )
    assert not history_reasons(base, require_name=False, lab="OpenAI")
    assert history_reasons(base, require_name=True, lab="OpenAI")
    assert "control_relationship_outside_scoped_non_affiliation" in history_reasons(
        base.replace("is separate from", "is not involved with"), require_name=False, lab="OpenAI"
    )


def tool_candidate():
    return {
        "family": "first_call",
        "tools": [
            {
                "function": {
                    "name": "f",
                    "parameters": {"type": "object", "required": ["x"], "properties": {"x": {"type": "integer"}}},
                }
            }
        ],
        "prompt": [{"role": "user", "content": "use x=2"}],
    }


def test_tool_calls_validate_original_schema_without_rewriting():
    row = tool_candidate()
    before = copy.deepcopy(row)
    assert not tool_reasons(row, '<tool_call>{"name":"f","arguments":{"x":2}}</tool_call>', "")
    assert "arguments_do_not_satisfy_original_schema" in tool_reasons(
        row, '<tool_call>{"name":"f","arguments":{}}</tool_call>', ""
    )
    assert row == before


def test_remote_schema_reference_rejects_without_fetching():
    row = tool_candidate()
    row["tools"][0]["function"]["parameters"] = {"$ref": "https://example.invalid/never-fetch"}
    assert "remote_schema_reference_not_resolved" in tool_reasons(
        row, '<tool_call>{"name":"f","arguments":{}}</tool_call>', ""
    )


def test_reply_shape_and_unverified_numeric_result_reject():
    row = tool_candidate()
    row["family"] = "reply"
    assert "reply_shape_emitted_another_call" in tool_reasons(
        row, '<tool_call>{"name":"f","arguments":{"x":2}}</tool_call>', ""
    )
    assert "unverified_numeric_result" in tool_reasons(row, "The result is 999.", "")


def test_cap2_is_per_exact_prompt_mode_full_row_not_answer_only():
    rows = []
    for sample in range(4):
        rows.append(
            {
                "id": str(sample),
                "candidate": {
                    "source_id": "q",
                    "group": "identity_v2",
                    "enable_thinking": False,
                    "sample_index": sample,
                },
                "decision": {"rejection_reasons": []},
                "student_prompt_token_ids": [1],
                "raw_target_token_ids": [2 if sample < 2 else sample + 2],
            }
        )
    kept, excluded = select(rows)
    assert [row["id"] for row in kept] == ["0", "2"]
    assert {tuple(row["reasons"]) for row in excluded} == {
        ("duplicate_full_student_row",),
        ("identity_prompt_mode_cap2",),
    }
