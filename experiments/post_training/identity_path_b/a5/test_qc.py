# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest

from experiments.post_training.identity_path_b.a5.data import TEACHER_SYSTEM, teacher_candidate
from experiments.post_training.identity_path_b.a5.qc import checked_student, text_flags
from experiments.post_training.identity_path_b.replay_capture import SOURCE_MODEL
from experiments.post_training.identity_t2.capture import request_body

FINAL = "I am Marin's Latest MoE. I was developed and trained by the Marin Community."


def test_teacher_and_student_context_preserve_user_order_without_Will_answers():
    off = {
        "id": "fixture-off",
        "source_id": "fixture",
        "family": "name",
        "messages": [
            {"role": "user", "content": "First user turn"},
            {"role": "assistant", "content": "EARLIER_WILL_ANSWER"},
            {"role": "user", "content": "Second user turn"},
            {"role": "assistant", "content": "FINAL_WILL_ANSWER"},
        ],
    }
    original = copy.deepcopy(off)
    row = teacher_candidate(off, 3)
    body = request_body(row, "native-v0.1")
    assert body["messages"] == [{"role": "system", "content": TEACHER_SYSTEM}, off["messages"][0], off["messages"][2]]
    assert row["prompt"] == [off["messages"][0], off["messages"][2]] and off == original
    assert "WILL_ANSWER" not in json.dumps(body)
    assert body["max_tokens"] == 8192 and body["temperature"] == 0.7 and body["top_p"] == 0.95
    assert body["n"] == 1 and body["return_token_ids"] and body["tool_choice"] == "none"


@pytest.mark.parametrize(
    "reasoning",
    [
        "The question asks my identity. I should give the model name and creator clearly.",
        "I am Marin's Latest MoE. Open Athena and Stanford maintain me; the Marin Community trained me.",
        "I am Marin's Latest MoE. I am not Gemini; the Marin Community trained me.",
    ],
)
def test_supported_Held_identity_and_role_scopes_pass_prescreen(reasoning):
    assert text_flags(reasoning, FINAL) == []


@pytest.mark.parametrize(
    "reasoning,final,expected",
    [
        ("The system prompt says to use that name.", FINAL, "omitted_system"),
        ("I was told I am Marin's Latest MoE.", FINAL, "omitted_system"),
        ("My instructions require that name.", FINAL, "omitted_system"),
        ("I am Gemini.", FINAL, "foreign_self"),
        ("Open Athena trained me.", FINAL, "profile_or_creator_role_mismatch"),
        ("I am Marin Snowball.", FINAL, "profile_or_creator_role_mismatch"),
        ("I am a transformer with 67 billion parameters.", FINAL, "unsupported"),
        ("<think>nested</think>", FINAL, "delimiter"),
        ("I should answer directly.", "I am Marin's Latest MoE.", "missing_positive_Marin_Community_creator"),
        ("I should answer directly.", "Open Athena and Stanford maintain me.", "missing_positive_Held_name"),
    ],
)
def test_QC_rejects_profile_leaks_unsupported_facts_and_missing_final_credits(reasoning, final, expected):
    assert any(expected in flag for flag in text_flags(reasoning, final))


def test_actual_request_sampling_change_is_rejected_before_output_admission():
    row = teacher_candidate(
        {
            "id": "fixture-off",
            "source_id": "fixture",
            "family": "name",
            "messages": [{"role": "user", "content": "Who are you?"}],
        },
        0,
    )
    body = request_body(row, SOURCE_MODEL)
    body["temperature"] = 0.0
    with pytest.raises(ValueError, match="Actual A5 request body"):
        checked_student({"candidate": row, "request": body, "fatal": "fixture_transport_failure"}, None)
