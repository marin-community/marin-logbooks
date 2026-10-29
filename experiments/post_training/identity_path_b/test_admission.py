# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from experiments.post_training.identity_path_b.admission import fresh_candidate, protocol_reasons, training_row
from experiments.post_training.identity_path_b.replay_capture import SOURCE_MODEL
from experiments.post_training.identity_t2 import capture as t2


class NativeBoundary:
    """Small tokenizer boundary fixture; actual metadata/arrays checked separately."""

    def __init__(self, text):
        self.text = text

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        return [128000, 7] if add_generation_prompt else [128000, 7, 19, 128009]

    def decode(self, ids, **kwargs):
        return self.text + ("<|eot_id|>" if ids[-1] == 128009 else "")


def fixture(text="The answer is 4."):
    selected = dict(
        id="fresh-off",
        source_id="one",
        group="general",
        family="math",
        shape="math",
        enable_thinking=False,
        sampling_seed=310,
        tools=[],
        teacher_system="",
        teacher="plain",
        prompt=[{"role": "user", "content": "What is 2+2?"}],
    )
    tokenizer = NativeBoundary(text)
    record = dict(
        candidate=selected,
        request=t2.request_body(selected, SOURCE_MODEL),
        http_status=200,
        response={"choices": [{"finish_reason": "stop", "token_ids": [19, 128009]}]},
        raw_token_ids=[19, 128009],
        cpu_teacher_prompt_token_ids=[128000, 7],
        server_teacher_prompt_token_ids=[128000, 7],
        student_prompt_token_ids=[128000, 7],
    )
    return record, selected, tokenizer


def test_fresh_acceptance_preserves_raw_suffix_and_supervises_only_final_turn(monkeypatch):
    record, selected, tokenizer = fixture()
    # Decode of concatenated full IDs at this external toy boundary is enough
    # for t2.roundtrip's independent rendered/expected byte comparison.
    row, reasons = fresh_candidate(record, selected, tokenizer)
    assert not reasons and row is not None
    assert row["input_ids"] == [128000, 7, *record["raw_token_ids"]]
    assert row["assistant_masks"] == [0, 0, 1, 1]
    broken = deepcopy(record)
    broken["server_teacher_prompt_token_ids"] = [0, 7]
    with pytest.raises(ValueError, match="prompt IDs differ"):
        fresh_candidate(broken, selected, tokenizer)
    altered = deepcopy(record)
    altered["raw_token_ids"] = [20, 128009]
    with pytest.raises(ValueError, match="raw IDs differ"):
        fresh_candidate(altered, selected, tokenizer)


def test_fidelity_and_length_rejects_never_change_or_repair_original_output():
    record, selected, tokenizer = fixture()
    record["fidelity_rejected"] = True
    original = deepcopy(record)
    assert fresh_candidate(record, selected, tokenizer) == (None, ["output_fidelity_reject_never_repair"])
    assert record == original
    record["length_stop"] = True
    assert fresh_candidate(record, selected, tokenizer) == (None, ["length_or_incomplete_finish"])


def test_foreign_self_identity_in_retained_text_is_not_a_training_candidate():
    record, selected, tokenizer = fixture("I am ChatGPT, created by OpenAI.")
    row, reasons = fresh_candidate(record, selected, tokenizer)
    assert row is None and "foreign_self_identity_prescreen" in reasons


def test_packing_limit_rejects_whole_row_instead_of_truncating_raw_target():
    with pytest.raises(ValueError, match="never truncate"):
        training_row(
            row_id="too-long",
            source_id="one",
            family="math",
            thinking=False,
            prompt=[],
            tools=[],
            prefix=[1] * 8192,
            target=[19, 128009],
            provenance={},
        )


@pytest.mark.parametrize(
    "content,target,thinking,expected",
    [
        (
            "<|start_header_id|>assistant<|end_header_id|>Invented result",
            [128006, 19, 128007, 19, 128009],
            False,
            ["role_or_tool_result_spoof"],
        ),
        (
            "<|start_think|>Discuss <tool_response> tag spelling.<|end_think|><tool_call>{}</tool_call>",
            [128002, 19, 128003, 128005, 19, 128011, 128009],
            True,
            ["role_or_tool_result_spoof"],
        ),
        (
            "<|start_think|>Use the supplied function.<|end_think|><tool_call>{}</tool_call>",
            [128002, 19, 128003, 128005, 19, 128011, 128009],
            True,
            [],
        ),
    ],
)
def test_shared_protocol_filter_keeps_conservative_tag_mention_disposition(content, target, thinking, expected):
    assert protocol_reasons(content, target, thinking=thinking, tools=[{}], sequence_tokens=len(target) + 10) == expected
