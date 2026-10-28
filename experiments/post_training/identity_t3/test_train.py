# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import haliax as hax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.data.text.datasets import ChatDataset
from levanter.models.lm_model import LmExample
from levanter.store.cache import CacheMetadata, SerialCacheWriter, TreeCache

from experiments.post_training.identity_t2.prepare import (
    SEQ_LEN,
    CapturedTargetProcessor,
    array_sha,
    check_validity,
    encoded_record,
    summarize,
)
from experiments.post_training.identity_t3.train import FrozenInputs, checked_batch, verify_arrays


@pytest.fixture
def native_batch(tmp_path):
    rows = []
    for index in range(8):
        rows.append(
            {
                "id": str(index),
                "student_prompt_token_ids": [128000, 128001, 42 + index, 128009],
                "raw_target_token_ids": [128005, 70 + index, 128011, 128009],
                "target_start": 4,
                "valid_target_length": 4,
                "decision": {"rejection_reasons": []},
                "candidate": {"group": "tools", "family": "first_call"},
                "raw_record_sha256": "fixture",
            }
        )
    processor = CapturedTargetProcessor()
    options = CacheMetadata(preprocessor_metadata=processor.metadata)
    path = str(tmp_path / "cache")
    with SerialCacheWriter(path, processor.output_exemplar, options) as writer:
        writer.write_batch(processor(rows))
    cache = TreeCache.load(path, processor.output_exemplar, options)
    dataset = ChatDataset(
        cache,
        hax.Axis("position", SEQ_LEN),
        max_segments_per_example=1,
        slice_strategy="raise",
        mask_user_turns=True,
        block_cross_document_attention=True,
    ).as_sync_dataset()
    examples = dataset.get_batch(list(range(8)))
    tokens = np.stack([np.asarray(example.tokens) for example in examples])
    weights = np.stack([np.asarray(example.loss_weight) for example in examples])
    query = np.stack([np.asarray(example.attn_mask.segment_ids[0]) for example in examples])
    keys = np.stack([np.asarray(example.attn_mask.segment_ids[1]) for example in examples])
    metadata = {row["id"]: encoded_record(row)[1] for row in rows}
    validity = {}
    for index, row in enumerate(rows):
        validity[row["id"]] = {
            **check_validity(query[index], keys[index], metadata[row["id"]]),
            "query_segments_sha256": array_sha(query[index]),
            "key_segments_sha256": array_sha(keys[index]),
        }
    ordered = [{"position": index, "update": 0, **metadata[row["id"]]} for index, row in enumerate(rows)]
    frozen = FrozenInputs(
        "fixture",
        "fixture",
        {"per_update": [summarize(ordered)]},
        {},
        ordered,
        metadata,
        {(row["tokens_sha256"], row["target_mask_sha256"]): key for key, row in metadata.items()},
        validity,
    )
    return tokens, weights, query, keys, frozen


def test_actual_native_batch_audit_accepts_valid_eos_and_excludes_padding(native_batch):
    tokens, weights, query, keys, frozen = native_batch
    result = verify_arrays(tokens, weights, query, keys, 0, frozen)
    assert result["ids"] == list(map(str, range(8)))
    assert result["supervised_tokens"] == 32
    assert all(
        row["actual_valid_query_positions"] == row["actual_valid_key_positions"] == 8 for row in result["validity"]
    )
    assert tokens[0, 1] == 128001 and query[0, 1] == keys[0, 1] == 0
    assert not np.any(weights[:, 8:]) and np.all(query[:, 8:] == -1)


def test_training_batch_without_explicit_segments_is_rejected(native_batch):
    tokens, _, _, _, frozen = native_batch
    batch_axis = hax.Axis("batch", 8)
    named = hax.named(jnp.asarray(tokens), (batch_axis, hax.Axis("position", SEQ_LEN)))
    batch = hax.vmap(LmExample.causal, batch_axis)(named)
    with pytest.raises(ValueError, match="dataset audit failed"):
        checked_batch(batch, 0, frozen)


@pytest.mark.parametrize("corruption", ["query_padding", "key_padding", "history_loss", "raw_target_ID", "batch_order"])
def test_training_audit_blocks_corruption_before_learning(native_batch, corruption):
    tokens, weights, query, keys, frozen = native_batch
    if corruption == "query_padding":
        query[:, 8:] = 0  # Historical defect: zero loss alone did not exclude dispatch padding.
    elif corruption == "key_padding":
        keys[:, 8:] = 0
    elif corruption == "history_loss":
        weights[0, 0] = 1
    elif corruption == "raw_target_ID":
        tokens[0, 4] = 42
    else:
        tokens[[0, 1]] = tokens[[1, 0]]
        weights[[0, 1]] = weights[[1, 0]]
    with pytest.raises((AssertionError, ValueError)):
        verify_arrays(tokens, weights, query, keys, 0, frozen)
