# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from experiments.post_training.identity_t2.census import census_row, foreign_evidence, invented_result_evidence


def row(answer, reasoning="", tool_messages=(), group="tools"):
    return {
        "id": "example",
        "candidate": {"group": group, "prompt": [{"role": "tool", "content": value} for value in tool_messages]},
        "decision": {"answer": answer, "reasoning": reasoning},
        "raw_record_sha256": "fixed",
    }


@pytest.mark.parametrize("text", ["As ChatGPT, I need to simulate it.", "I am Claude.", "My creator is Google."])
def test_v4r2_foreign_self_identity_is_checked_in_retained_reasoning(text):
    record = row('<tool_call>{"name":"f","arguments":{}}</tool_call>', text)
    original = copy.deepcopy(record)
    assert census_row(record)["exclude"]
    assert record == original


def test_negated_foreign_statement_keeps_the_existing_predicate_semantics():
    assert not foreign_evidence("I am not ChatGPT.")


def test_same_reasoning_foreign_census_applies_to_controls():
    assert census_row(row("A factual answer.", "As ChatGPT, I should explain.", group="third_person"))["exclude"]


def test_fabricated_tool_role_news_payload_without_source_tool_message_rejects():
    record = row(
        '<tool_call>{"name":"get_news","arguments":{}}</tool_call>'
        '<|start_header_id|>tool<|end_header_id|><tool_response>{"news":[{"title":"Invented"}]}</tool_response>'
    )
    assert any(hit["class"] == "invented_tool_role_payload" for hit in invented_result_evidence(record))


def test_exact_existing_tool_payload_is_not_a_new_unmatched_result():
    payload = '{"news":[{"title":"Observed"}]}'
    assert not invented_result_evidence(row("<tool_response>" + payload + "</tool_response>", tool_messages=[payload]))


def test_pretool_result_assertion_is_not_supplied_by_user_arguments():
    assert invented_result_evidence(row("The monthly loan payment is approximately $551.55."))
    assert not invented_result_evidence(
        row("The monthly loan payment is $551.55.", tool_messages=['{"payment":551.55}'])
    )


def test_planned_call_arguments_are_not_misread_as_nested_result_payload():
    reasoning = 'The call should be {"name":"set_status","arguments":{"status":"active"}}.'
    assert not invented_result_evidence(
        row('<tool_call>{"name":"set_status","arguments":{"status":"active"}}</tool_call>', reasoning)
    )
