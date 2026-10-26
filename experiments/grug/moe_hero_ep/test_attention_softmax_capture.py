# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh
from levanter.grug.attention import AttentionMask, attention

from experiments.grug.moe_hero_ep.attention_softmax_capture import (
    attention_finish,
    attention_inputs,
    selected_attention_vjp,
)
from experiments.grug.moe_hero_ep.model import Block, CausalSelfAttention, GrugModelConfig


def _configuration():
    return GrugModelConfig(
        vocab_size=32,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=1,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        capacity_factor=4.0,
        initializer_std=0.5 / math.sqrt(32),
        qk_mult=1.3,
        qk_mult_long_scale=1.2,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
        rope_fused=True,
        sconv=True,
    )


def _mesh():
    return Mesh(
        np.asarray(jax.devices()).reshape(1, 1, 1, len(jax.devices()), 1),
        ("replica_dcn", "data", "context", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )


@pytest.mark.parametrize("is_global", [False, True])
def test_attention_capture_split_matches_production_attention_output_and_input_gradient(is_global):
    mesh = _mesh()
    cfg = _configuration()
    rows = 2 * len(jax.devices())
    x = jax.random.normal(jax.random.key(71), (rows, 8, cfg.hidden_dim))
    dy = jax.random.normal(jax.random.key(72), x.shape)
    segment_ids = jnp.broadcast_to(jnp.array([0, 0, 0, 1, 1, 1, 1, 1]), (rows, 8))
    mask = AttentionMask.causal(sliding_window=None if is_global else 4).with_segment_ids(segment_ids)
    flag = jnp.asarray(is_global)
    with jax.set_mesh(mesh):
        module = CausalSelfAttention.init(cfg, key=jax.random.key(73))

        def split(value):
            q, k, v = attention_inputs(module, value, mask, is_global=flag)
            output = attention(q, k, v, mask, implementation="reference")
            return attention_finish(module, value, output, v)

        direct, direct_pullback = jax.vjp(lambda value: module(value, mask, flag, flag), x)
        captured, split_pullback = jax.vjp(split, x)
        direct_dx = direct_pullback(dy)[0]
        split_dx = split_pullback(dy)[0]
    np.testing.assert_allclose(captured, direct, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(split_dx, direct_dx, rtol=1e-5, atol=1e-5)


def test_attention_capture_downstream_cotangent_reproduces_complete_block_query_weight_gradient():
    mesh = _mesh()
    cfg = _configuration()
    rows = 2 * len(jax.devices())
    x = jax.random.normal(jax.random.key(74), (rows, 8, cfg.hidden_dim))
    dy = jax.random.normal(jax.random.key(75), x.shape)
    mask = AttentionMask.causal(sliding_window=4)
    flag = jnp.asarray(False)
    with jax.set_mesh(mesh):
        block = Block.init(cfg, key=jax.random.key(76))
        capture = selected_attention_vjp(block, x, dy, mask, flag)
        reference_output = block(x, mask, flag, flag)[0]

        def complete_loss(w_q):
            updated = eqx.tree_at(lambda value: value.attn.w_q, block, w_q)
            return jnp.sum(updated(x, mask, flag, flag)[0] * dy)

        direct = jax.grad(complete_loss)(block.attn.w_q)
        attn_in = block.attn_gated_norm(block.rms_attn(x))

        def raw_output(w_q):
            updated = eqx.tree_at(lambda value: value.w_q, block.attn, w_q)
            q, k, v = attention_inputs(updated, attn_in, mask, is_global=flag)
            return attention(q, k, v, mask, implementation="reference")

        _, pullback = jax.vjp(raw_output, block.attn.w_q)
        reconstructed = pullback(capture.dout)[0]
    np.testing.assert_allclose(reconstructed, direct, rtol=1e-5, atol=1e-5)
    assert float(capture.tail_output_max_abs_error) < 1e-5
    np.testing.assert_allclose(
        capture.tail_output_reference_rms,
        jnp.sqrt(jnp.mean(jnp.square(reference_output))),
        rtol=1e-5,
        atol=1e-5,
    )
