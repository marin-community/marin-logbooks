# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import io

import equinox as eqx
import haliax as hax
import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from levanter.data.loader import stack_tree
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import (
    SnowballConfig,
    SnowballLMHeadModel,
    dispatch_token_validity,
    snowball_loss_with_dispatch_stats,
)

from experiments.post_training.identity_path_b.native import packed_dataset


def test_actual_packed_parquet_reader_preserves_document_attention_and_shifted_loss():
    batch, position = hax.Axis("batch", 2 * jax.device_count()), hax.Axis("position", 12)
    records = []
    for segmented in (True, False):
        for changed in (False, True):
            tokens = ([11, 12, 13, 14] if changed else [1, 2, 3, 4]) + [5, 6, 7, 8, 9, 10, 0, 0]
            row = {
                "input_ids": tokens,
                "target_weights": [0.0] * 5 + [1.0] * 5 + [0.0] * 2,
                "segment_ids": ([0] * 4 + [1] * 6 if segmented else [0] * 10) + [-1] * 2,
            }
            records.extend([row] * batch.size)
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(records), buffer)
    raw = buffer.getvalue()
    dataset = packed_dataset(raw, hashlib.sha256(raw).hexdigest(), position)
    examples = [
        stack_tree(batch, asyncio.run(dataset.get_batch(list(range(index, index + batch.size)))))
        for index in range(0, len(records), batch.size)
    ]
    for example in examples:
        assert example.attn_mask.segment_ids[1].axes[-1].name == "key_position"
        np.testing.assert_array_equal(
            example.loss_weight.array, np.broadcast_to([0] * 4 + [1] * 5 + [0] * 3, (batch.size, 12))
        )
    config = SnowballConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=16,
        sliding_window=4,
        initializer_std=0.1,
        attention_implementation="torch_order",
    )
    with (
        jax.set_mesh(compact_grug_mesh(expert_axis_size=1)),
        hax.axis_mapping({"batch": ("replica_dcn", "data", "expert"), "vocab": "model"}),
    ):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", 32), config, key=jax.random.PRNGKey(310))
        examples = [hax.auto_sharded(example) for example in examples]
        forward, checked = eqx.filter_jit(model.activations_with_stats), eqx.filter_jit(
            snowball_loss_with_dispatch_stats
        )
        hidden, losses = [], []
        for example in examples:
            value, stats = forward(example.tokens, dispatch_token_validity(example), attn_mask=example.attn_mask)
            np.testing.assert_array_equal(stats.dropped, np.zeros(2))
            hidden.append(np.asarray(value.array[:, 4:10]))
            losses.append(np.asarray(checked(model, example)[0].array))
        np.testing.assert_array_equal(hidden[0], hidden[1])
        np.testing.assert_array_equal(losses[0], losses[1])
        assert np.max(np.abs(hidden[2] - hidden[3])) > 1e-6
        assert abs(float(losses[2]) - float(losses[3])) > 1e-6
        ordinary = eqx.filter_jit(model.compute_next_token_loss)(examples[0])
        np.testing.assert_allclose(losses[0], ordinary.array, rtol=1e-5, atol=1e-5)
