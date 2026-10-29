# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import replace

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.data.text.datasets import ChatDataset
from levanter.data.text.examples import GrugLmExample, named_lm_example_from_grug
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.grug.sharding import compact_grug_mesh
from levanter.layers.attention_mask import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.snowball import (
    SnowballConfig,
    SnowballLMHeadModel,
    dispatch_token_validity,
    snowball_loss_with_dispatch_stats,
)
from levanter.store.cache import SerialCacheWriter
from levanter.testing.cpu_devices import run_on_cpu_devices


def test_unpacked_chat_validity_keeps_prompt_and_real_eos(tmp_path):
    row = {
        "input_ids": np.array([128000, 31, 32, 128001], np.int32),
        "assistant_masks": np.array([0, 0, 1, 1], np.int32),
    }
    exemplar = {name: np.zeros((0,), np.int32) for name in row}
    with SerialCacheWriter(str(tmp_path / "cache"), exemplar) as writer:
        writer.write_batch([row])
    position = hax.Axis("position", 8)
    dataset = ChatDataset(
        writer.result(),
        position,
        max_segments_per_example=1,
        slice_strategy="raise",
        block_cross_document_attention=True,
    )
    example = named_lm_example_from_grug(asyncio.run(dataset.getitem_async(0)), position)
    np.testing.assert_array_equal(dispatch_token_validity(example).array, [True] * 4 + [False] * 4)
    segments = example.attn_mask.segment_ids[0].array
    assert np.all(segments[:4] >= 0)
    np.testing.assert_array_equal(segments[4:], -np.ones(4))
    assert example.tokens.array[3] == 128001
    assert example.loss_weight.array[0] == 0
    same_token_padding = eqx.tree_at(
        lambda value: value.tokens, example, hax.named(jnp.array([128000, 31, 32] + [128001] * 5), (position,))
    )
    np.testing.assert_array_equal(
        dispatch_token_validity(same_token_padding).array, dispatch_token_validity(example).array
    )
    with pytest.raises(ValueError, match="segment IDs"):
        dispatch_token_validity(LmExample.causal(example.tokens))


def test_stats_forward_preserves_observer_and_loss():
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
        attention_implementation="reference",
    )
    batch = hax.Axis("batch", 2 * jax.device_count())
    position = hax.Axis("position", 8)
    ids = hax.named(jnp.broadcast_to(jnp.arange(8), (batch.size, 8)), (batch, position))
    valid = hax.named(jnp.ones(ids.array.shape, dtype=jnp.bool_), ids.axes)
    with (
        jax.set_mesh(compact_grug_mesh()),
        hax.axis_mapping({"batch": ("replica_dcn", "data", "expert"), "vocab": "model"}),
    ):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", 32), config, key=jax.random.PRNGKey(310))
        hidden, routes = eqx.filter_jit(model.activations_with_routes)(ids)
        measured, stats = eqx.filter_jit(model.activations_with_stats)(ids, valid)
        np.testing.assert_array_equal(measured.array, hidden.array)
        np.testing.assert_array_equal(stats.routes, routes)
        np.testing.assert_array_equal(stats.dropped, np.zeros(2))
        example = hax.vmap(LmExample.causal, batch)(
            ids, segment_ids=hax.named(jnp.zeros(ids.array.shape, jnp.int32), ids.axes)
        )
        example = hax.auto_sharded(example)
        loss, metrics = eqx.filter_jit(snowball_loss_with_dispatch_stats)(model, example)
        reference = eqx.filter_jit(model.compute_next_token_loss)(example)
        np.testing.assert_allclose(loss.array, reference.array, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(metrics["moe_dropped"], stats.dropped)


@pytest.mark.parametrize("implementation", ["reference", "torch_order"])
@pytest.mark.timeout(120)
def test_packed_documents_isolate_real_hidden_states_and_target_loss(implementation):
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
        attention_implementation=implementation,
    )  # First layer is short/RoPE; the last is long/NoPE.
    batch, position = hax.Axis("batch", 2 * jax.device_count()), hax.Axis("position", 12)
    first = jnp.broadcast_to(jnp.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0]), (batch.size, 12))
    changed = first.at[:, :4].set(jnp.asarray([11, 12, 13, 14]))
    segments = hax.named(jnp.broadcast_to(jnp.asarray([0] * 4 + [1] * 6 + [-1] * 2), first.shape), (batch, position))
    control_segments = hax.where(segments >= 0, 0, -1)
    # Only docB query positions that predict another docB token carry loss.
    weights = hax.named(jnp.broadcast_to(jnp.asarray([0] * 4 + [1] * 5 + [0] * 3), first.shape), (batch, position))
    ids = [hax.named(value, (batch, position)) for value in (first, changed)]
    examples = [
        named_lm_example_from_grug(
            jax.vmap(GrugLmExample.causal)(value.array, loss_weight=weights.array, segment_ids=segments.array),
            position,
            batch,
        )
        for value in ids
    ]
    controls = [
        named_lm_example_from_grug(
            jax.vmap(GrugLmExample.causal)(value.array, loss_weight=weights.array, segment_ids=control_segments.array),
            position,
            batch,
        )
        for value in ids
    ]
    assert examples[0].attn_mask.segment_ids[1].axes[-1].name == "key_position"
    single = hax.vmap(LmExample.causal, batch)(
        ids[0], segment_ids=hax.named(jnp.zeros(first.shape, jnp.int32), ids[0].axes)
    )
    with (
        jax.set_mesh(compact_grug_mesh(expert_axis_size=1)),
        hax.axis_mapping({"batch": ("replica_dcn", "data", "expert"), "vocab": "model"}),
    ):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", 32), config, key=jax.random.PRNGKey(310))
        forward = eqx.filter_jit(model.activations_with_stats)
        measured = [forward(value, segments >= 0, attn_mask=examples[0].attn_mask) for value in ids]
        for _, stats in measured:
            np.testing.assert_array_equal(stats.dropped, np.zeros(2))
        np.testing.assert_array_equal(measured[0][0].array[:, 4:10], measured[1][0].array[:, 4:10])
        examples = [hax.auto_sharded(example) for example in examples]
        controls = [hax.auto_sharded(example) for example in controls]
        checked = eqx.filter_jit(snowball_loss_with_dispatch_stats)
        losses = [checked(model, example)[0].array for example in examples]
        np.testing.assert_array_equal(losses[0], losses[1])
        ordinary = eqx.filter_jit(model.compute_next_token_loss)(examples[0])
        np.testing.assert_allclose(losses[0], ordinary.array, rtol=1e-5, atol=1e-5)
        unsegmented = [forward(value, segments >= 0, attn_mask=controls[0].attn_mask) for value in ids]
        difference = np.max(np.abs(np.asarray(unsegmented[0][0].array[:, 4:10] - unsegmented[1][0].array[:, 4:10])))
        assert difference > 1e-6
        control_losses = [checked(model, example)[0].array for example in controls]
        assert abs(float(control_losses[0]) - float(control_losses[1])) > 1e-6
        all_valid = hax.named(jnp.ones(first.shape, dtype=jnp.bool_), ids[0].axes)
        no_mask = forward(ids[0], all_valid)[0].array
        one_document = forward(ids[0], all_valid, attn_mask=single.attn_mask)[0].array
        np.testing.assert_array_equal(no_mask, one_document)
        np.testing.assert_array_equal(eqx.filter_jit(model.activations)(ids[0]).array, no_mask)
        unsupported = [
            AttentionMask(is_causal=False),
            replace(single.attn_mask, explicit_mask=all_valid),
            replace(single.attn_mask, causal_offset=hax.named(jnp.asarray(0), ())),
            replace(single.attn_mask, sliding_window=2),
            replace(single.attn_mask, bidirectional_window=2),
            all_valid,
        ]
        document_mask = examples[0].attn_mask
        assert isinstance(document_mask, AttentionMask) and document_mask.segment_ids is not None
        query, keys = document_mask.segment_ids
        distinct_keys = keys + 7
        converted = model._attention_mask(ids[0], replace(document_mask, segment_ids=(query, distinct_keys)))
        assert converted is not None and converted.segment_ids is not None
        np.testing.assert_array_equal(converted.segment_ids[0], query.array)
        np.testing.assert_array_equal(converted.segment_ids[1], distinct_keys.array)
        bad_keys = hax.named(jnp.zeros((batch.size, 11), jnp.int32), (batch, hax.Axis("key_position", 11)))
        unsupported.append(replace(document_mask, segment_ids=(query, bad_keys)))
        for mask in unsupported:
            with pytest.raises(ValueError, match="Snowball"):
                forward(ids[0], all_valid, attn_mask=mask)
        raw_masks = [
            GrugAttentionMask(is_causal=False),
            GrugAttentionMask.causal(sliding_window=2),
            GrugAttentionMask.causal().with_fa4_bounds(jnp.zeros(first.shape, jnp.int32), all_valid.array),
        ]
        for mask in raw_masks:
            with pytest.raises(ValueError, match="Snowball transformer"):
                eqx.filter_jit(model.transformer.with_stats)(ids[0].array, all_valid.array, mask)


@pytest.mark.timeout(240)
def test_dropless_bos_routes_ignore_companions_and_padding():
    run_on_cpu_devices(
        """
        from dataclasses import replace
        import equinox as eqx
        import haliax as hax
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh
        from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel

        mesh = Mesh(np.array(jax.devices()).reshape(1, 2, 4, 1, 1, 1),
            ('replica_dcn', 'data', 'expert', 'context', 'replica', 'model'),
            axis_types=(AxisType.Explicit,) * 6)
        config = SnowballConfig(vocab_size=32, hidden_dim=16, intermediate_dim=16,
            shared_expert_intermediate_dim=16, num_experts=8, num_experts_per_token=2,
            num_layers=3, num_heads=2, num_kv_heads=1, head_dim=8,
            max_seq_len=32, sliding_window=4, attention_implementation='reference',
            moe_implementation='ring_gather_combine')
        with jax.set_mesh(mesh):
            model = SnowballLMHeadModel.init(hax.Axis('vocab', 32), config, key=jax.random.PRNGKey(310))
            blocks = tuple(replace(block, mlp=replace(block.mlp,
                expert_mlp=replace(block.mlp.expert_mlp, capacity_factor=4.0)))
                for block in model.transformer.blocks)
            model = replace(model, transformer=replace(model.transformer, blocks=blocks))
            forward = eqx.filter_jit(model.activations_with_stats)
            baseline = None
            for sequence, offset in [(8, 0), (16, 7)]:
                tokens = (np.arange(8 * sequence).reshape(8, sequence) + offset) % 32
                tokens[:, 0] = 1
                valid = np.arange(sequence)[None, :] < np.array([5, 4, 3, 6, 2, 7, 4, 5])[:, None]
                tokens[~valid] = 2
                axes = (hax.Axis('batch', 8), hax.Axis('position', sequence))
                _, stats = forward(hax.named(jnp.asarray(tokens), axes), hax.named(jnp.asarray(valid), axes))
                np.testing.assert_array_equal(stats.dropped, np.zeros(3))
                bos = np.asarray(stats.routes)[:, :, 0, :]
                np.testing.assert_array_equal(bos, np.broadcast_to(bos[:, :1, :], bos.shape))
                if baseline is not None:
                    np.testing.assert_array_equal(bos, baseline)
                baseline = bos
            capped_blocks = tuple(replace(block, mlp=replace(block.mlp,
                router=jnp.zeros_like(block.mlp.router),
                router_bias=jnp.arange(8, 0, -1, dtype=jnp.float32),
                expert_mlp=replace(block.mlp.expert_mlp, capacity_factor=1.0)))
                for block in model.transformer.blocks)
            capped = replace(model, transformer=replace(model.transformer, blocks=capped_blocks))
            axes = (hax.Axis('batch', 8), hax.Axis('position', 8))
            tokens = hax.named(jnp.ones((8, 8), jnp.int32), axes)
            valid = hax.named(jnp.broadcast_to(jnp.arange(8) < 4, (8, 8)), axes)
            _, counts = eqx.filter_jit(capped.activations_with_stats)(tokens, valid)
            np.testing.assert_array_equal(counts.sender_dropped, np.zeros(3))
            np.testing.assert_array_equal(counts.receiver_dropped, np.full(3, 48))
            np.testing.assert_array_equal(counts.dropped, np.full(3, 48))
        """,
        device_count=8,
    )
