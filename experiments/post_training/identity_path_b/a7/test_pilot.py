# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest
import requests

from experiments.post_training.identity_path_b.a7 import capture
from experiments.post_training.identity_path_b.a7.data import TEACHER, request_body
from experiments.post_training.identity_path_b.a7.qc import checked_student, positive_name, text_flags

FINAL = "I am Marin's Latest MoE. I was developed and trained by the Marin Community."


@pytest.mark.parametrize(
    "reasoning",
    [
        "The system says that name.",
        "The systems suggest that name.",
        "According to policy, use that name.",
        "The policies say this.",
        "The instruction says this.",
        "The instructions say this.",
        "The prompt asks this.",
        "The prompts ask this.",
    ],
)
def test_literal_reasoning_references_rejected_even_when_provider_reasoning_is_empty(reasoning):
    assert "reasoning:literal_system_policy_instruction_prompt" in text_flags(reasoning, FINAL)


def test_supported_identity_and_creator_wording_still_passes():
    assert text_flags("The question asks my identity. I should give the model name and creator clearly.", FINAL) == []
    joint = "I am Marin's Latest MoE. The **Marin Community** develops, trains, and maintains the model."
    assert "final_missing_positive_Marin_Community_creator" not in text_flags("I should answer clearly.", joint)
    assert any("foreign_self" in flag for flag in text_flags("I am Gemini.", FINAL))


def test_no_system_or_name_instruction_added_and_sampling_admission_is_exact():
    row = dict(
        prompt=[dict(role="user", content="Who maintains you?")],
        teacher_system="",
        tools=[],
        enable_thinking=True,
        sampling_seed=17,
    )
    body = request_body(row)
    assert body["model"] == TEACHER and body["messages"] == row["prompt"]
    assert body["temperature"] == 0.7 and body["top_p"] == 0.95 and body["max_tokens"] == 8192 and body["n"] == 1
    assert body["add_generation_prompt"] is True and body["continue_final_message"] is False
    changed = copy.deepcopy(body)
    changed["seed"] = 18
    with pytest.raises(ValueError, match="Actual A7 request"):
        checked_student(dict(candidate=row, request=changed, fatal="transport"), None)


def test_bad_server_prefill_prefix_prevents_completion_and_preserves_failed_record(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(capture, "teacher_prompt_ids", lambda t, r: [11, 128002])
    monkeypatch.setattr(capture.t2, "prompt_ids", lambda t, r: [11])

    def post(url, **kwargs):
        calls.append((url, kwargs["json"]))
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"tokens":[11]}'
        return response

    monkeypatch.setattr(requests, "post", post)
    row = dict(
        prompt=[dict(role="user", content="Who are you?")],
        teacher_system="",
        tools=[],
        enable_thinking=True,
        sampling_seed=17,
    )
    record = capture.capture_one(
        0, row, endpoint="https://fixture.invalid/v1", model=TEACHER, tokenizer=None, out=tmp_path
    )
    assert record["fatal"] and len(calls) == 1 and calls[0][0].endswith("/tokenize")
    assert calls[0][1]["add_generation_prompt"] is True and calls[0][1]["continue_final_message"] is False
    assert json.loads((tmp_path / "0000.json").read_text()) == record


@pytest.mark.parametrize(
    "final",
    [
        "Marin\u2019s Latest MoE, developed and trained by the Marin Community.",
        "**Marin\u2019s Latest MoE**, developed and trained by the **Marin Community**.",
        "Marin's Latest MoE is developed and trained by the Marin Community.",
        "I am **Marin\u2019s Latest MoE**. I was developed and trained by the **Marin Community**.",
        "I am Marin's Latest MoE, not ChatGPT. I was trained by the Marin Community.",
    ],
)
def test_affirmative_name_fragments_and_markup_are_predicate_only(final):
    before = final
    assert positive_name(final)
    assert text_flags("The question asks my identity.", final) == []
    assert final == before


@pytest.mark.parametrize(
    "final",
    [
        "Marin's Latest MoE is not my name. I was trained by the Marin Community.",
        "Marin's Latest MoE is another model, not me. I was trained by the Marin Community.",
        "Marin's Latest MoE is ChatGPT. I was trained by the Marin Community.",
        "If I were asked, I could say I am Marin's Latest MoE. I was trained by the Marin Community.",
        'They said: "I am Marin\'s Latest MoE. I was trained by the Marin Community."',
        "Marin's Latest MoE, developed by OpenAI.",
        "I am Marin's Latest MoE, hypothetically. I was trained by the Marin Community.",
        "`I am Marin's Latest MoE.` I was trained by the Marin Community.",
        "I am **Gemini**. The Marin Community trained me.",
        "I am Marin's Latest MoE. The Marin Community contributed to my training.",
        "I am **Marin's Latest MoE**. I was developed by **OpenAI** and maintained by the Marin Community.",
        "I am Marin's Latest MoE. The Marin Community maintains me.",
        "I am Marin's Latest MoE. I was not developed or trained by the Marin Community.",
    ],
)
def test_quoted_hypothetical_negated_foreign_and_maintainer_wording_stays_rejected(final):
    assert text_flags("The question asks my identity.", final)
