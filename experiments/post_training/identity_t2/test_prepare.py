# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy

import haliax as hax
import numpy as np
import pytest
from levanter.data.text.datasets import ChatDataset
from levanter.store.cache import CacheMetadata, SerialCacheWriter, TreeCache

from experiments.post_training.identity_t2.prepare import (
    NEW_ROW_SAMPLING_WEIGHTS,
    SEQ_LEN,
    CapturedTargetProcessor,
    check_validity,
    encoded_record,
    summarize,
)


def captured_row():
    prefix = [128000, 42, 128005, 43, 128011, 128009, 44]
    target = [128005, 45, 128011, 128009]
    return {
        "id": "raw-example",
        "student_prompt_token_ids": prefix,
        "raw_target_token_ids": target,
        "decision": {"rejection_reasons": []},
        "target_start": len(prefix),
        "valid_target_length": len(target),
        "candidate": {"group": "tools", "family": "first_call"},
        "raw_record_sha256": "bound-raw-hash",
    }


def native_example(row, tmp_path):
    processor = CapturedTargetProcessor()
    options = CacheMetadata(preprocessor_metadata=processor.metadata)
    with SerialCacheWriter(str(tmp_path), processor.output_exemplar, options) as writer:
        writer.write_batch(processor([row]))
    cache = TreeCache.load(str(tmp_path), processor.output_exemplar, options)
    dataset = ChatDataset(
        cache,
        hax.Axis("position", SEQ_LEN),
        max_segments_per_example=1,
        slice_strategy="raise",
        mask_user_turns=True,
        block_cross_document_attention=True,
    )
    return dataset.as_sync_dataset().get_batch([0])[0]


def test_raw_ids_history_and_variable_length_cache_are_preserved():
    row = captured_row()
    before = copy.deepcopy(row)
    document, metadata = encoded_record(row)
    full = row["student_prompt_token_ids"] + row["raw_target_token_ids"]
    assert document["input_ids"].tolist() == full
    assert len(document["input_ids"]) == metadata["valid_input_length"] == len(full)
    assert not np.any(document["assistant_masks"][: len(row["student_prompt_token_ids"])])
    assert document["assistant_masks"].sum() == len(row["raw_target_token_ids"])
    assert row == before


def test_actual_native_chat_path_shifts_mask_once_and_marks_padding_invalid(tmp_path):
    row = captured_row()
    _, metadata = encoded_record(row)
    example = native_example(row, tmp_path / "cache")
    full = row["student_prompt_token_ids"] + row["raw_target_token_ids"]
    assert np.asarray(example.tokens)[: len(full)].tolist() == full
    expected_mask = np.zeros(SEQ_LEN)
    expected_mask[len(row["student_prompt_token_ids"]) - 1 : len(full) - 1] = 1
    np.testing.assert_array_equal(np.asarray(example.loss_weight), expected_mask)
    assert set(np.unique(np.asarray(example.loss_weight))) == {0, 1}
    check_validity(*example.attn_mask.segment_ids, metadata)
    assert NEW_ROW_SAMPLING_WEIGHTS["tools"] == 2.0  # sampling only; binary loss weights above


def test_valid_eos_valued_token_remains_valid_inside_one_native_document(tmp_path):
    row = captured_row()
    row["student_prompt_token_ids"][1] = 128001
    document, metadata = encoded_record(row)
    assert document["input_ids"][1] == 128001
    example = native_example(row, tmp_path / "cache")
    check_validity(*example.attn_mask.segment_ids, metadata)
    query, key = example.attn_mask.segment_ids
    assert query[1] >= 0 and key[1] >= 0
    assert len(np.unique(np.asarray(query)[: metadata["valid_input_length"]])) == 1


def test_420_token_regression_does_not_dispatch_7772_padding_positions(tmp_path):
    row = captured_row()
    row["student_prompt_token_ids"] = [42] * 416
    row["target_start"] = 416
    _, metadata = encoded_record(row)
    example = native_example(row, tmp_path / "cache")
    proof = check_validity(*example.attn_mask.segment_ids, metadata)
    assert proof["actual_valid_query_positions"] == proof["actual_valid_key_positions"] == 420
    assert np.all(np.asarray(example.attn_mask.segment_ids[0])[420:] == -1)


def test_validity_check_rejects_old_zero_padded_prebuilt_segments():
    _, metadata = encoded_record(captured_row())
    with pytest.raises(AssertionError):
        check_validity(np.zeros(SEQ_LEN), np.zeros(SEQ_LEN), metadata)


def test_oversize_inputs_are_not_sliced_or_repaired():
    row = captured_row()
    row["student_prompt_token_ids"] = [42] * SEQ_LEN
    row["target_start"] = SEQ_LEN
    with pytest.raises(ValueError, match="exceeds"):
        encoded_record(row)


def test_rejected_row_cannot_enter_candidate_cache():
    row = captured_row()
    row["decision"]["rejection_reasons"] = ["human_reject"]
    with pytest.raises(AssertionError):
        encoded_record(row)


def test_share_counts_old_and_new_identity_tokens_not_row_frequency():
    result = summarize(
        [
            {"id": "a", "component": "identity", "class_tokens": {"identity_off": 3}},
            {"id": "b", "component": "identity_v2", "class_tokens": {"identity_v2": 2}},
            {"id": "c", "component": "preservation", "class_tokens": {"replay_chat": 95}},
        ]
    )
    assert result["identity_tokens"] == 5 and result["supervised_tokens"] == 100
    assert result["identity_token_share"] == 0.05
