# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sparse-query replay through the actual SM100 fused attention backward."""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from levanter.grug.attention._fa4_cute_backend import fa4_cute_attention_forward
from levanter.grug.attention._fa4_cute_config import sm100_flash4_cute_kernel_config

from experiments.grug.moe_hero_ep.attention_softmax_capture import AttentionCapture

BATCH_AXES = ("replica_dcn", "data", "expert")
QUERY_SELECTION_SEED = 1729


class AttentionPanelSelection(NamedTuple):
    q: jax.Array
    k: jax.Array
    v: jax.Array
    sparse_dout: jax.Array
    original_output: jax.Array
    positions: jax.Array
    lower_bounds: jax.Array
    valid: jax.Array
    kv_group: jax.Array
    local_row: jax.Array
    selection_fallback_count: jax.Array


class AttentionReplay(NamedTuple):
    q: jax.Array
    k: jax.Array
    v: jax.Array
    dout: jax.Array
    output: jax.Array
    dq: jax.Array
    dk: jax.Array
    dv: jax.Array
    positions: jax.Array
    lower_bounds: jax.Array
    valid: jax.Array
    kv_group: jax.Array
    local_row: jax.Array
    replay_output_max_abs_error: jax.Array
    selection_fallback_count: jax.Array


def select_attention_panel(
    capture: AttentionCapture, *, rank: jax.Array, layer_index: int, query_count: int
) -> AttentionPanelSelection:
    """Select one local sequence/KV group and stratified multi-key query rows.

    Every key position is retained. A stratum without a valid multi-key row falls
    back to a valid single-key row, or an invalid placeholder if none is valid.
    These fallbacks remain explicit so numerical summaries can exclude them.
    """
    batch, length, q_heads, width = capture.q.shape
    kv_heads = capture.k.shape[2]
    if q_heads % kv_heads or length % query_count:
        raise ValueError("Query heads must divide KV groups and sequence length must divide query strata")
    ratio = q_heads // kv_heads
    row = rank % batch
    group = (rank + layer_index) % kv_heads

    def query_group(value):
        return jax.lax.dynamic_slice(value, (row, 0, group * ratio, 0), (1, length, ratio, width))

    def key_group(value):
        return jax.lax.dynamic_slice(value, (row, 0, group, 0), (1, length, 1, value.shape[-1]))

    q = query_group(capture.q)
    k = key_group(capture.k)
    v = key_group(capture.v)
    output = query_group(capture.output)
    dout = query_group(capture.dout)
    lower = jax.lax.dynamic_slice(capture.lower_bounds, (row, 0), (1, length))
    valid = jax.lax.dynamic_slice(capture.valid, (row, 0), (1, length))
    stratum_width = length // query_count
    candidate_positions = jnp.arange(length).reshape(query_count, stratum_width)
    candidate_valid = valid.reshape(query_count, stratum_width)
    multiple_keys = candidate_valid & (candidate_positions - lower.reshape(query_count, stratum_width) >= 1)
    key = jax.random.fold_in(jax.random.PRNGKey(QUERY_SELECTION_SEED), layer_index)
    key = jax.random.fold_in(key, rank)
    scores = jax.random.uniform(key, (query_count, stratum_width))
    eligible_choice = jnp.argmin(jnp.where(multiple_keys, scores, jnp.inf), axis=-1)
    valid_choice = jnp.argmin(jnp.where(candidate_valid, scores, jnp.inf), axis=-1)
    has_multiple = jnp.any(multiple_keys, axis=-1)
    has_valid = jnp.any(candidate_valid, axis=-1)
    selected = jnp.where(has_multiple, eligible_choice, jnp.where(has_valid, valid_choice, 0))
    positions = jnp.arange(query_count) * stratum_width + selected
    sparse_dout = jnp.zeros_like(dout).at[:, positions].set(dout[:, positions])
    return AttentionPanelSelection(
        q,
        k,
        v,
        sparse_dout,
        output,
        positions,
        lower,
        valid,
        group,
        row,
        jnp.sum(~has_multiple, dtype=jnp.int32),
    )


def _replay_local(capture: AttentionCapture, *, layer_index: int, query_count: int) -> AttentionReplay:
    rank = jax.lax.axis_index(BATCH_AXES)
    selected = select_attention_panel(capture, rank=rank, layer_index=layer_index, query_count=query_count)
    if selected.q.shape[2] != 4 or selected.q.shape[-1] != 128:
        raise ValueError("The native Hero attention panel requires GQA4 and head dimension 128")
    config = sm100_flash4_cute_kernel_config()

    def fused(q, k, v):
        return fa4_cute_attention_forward(
            q,
            k,
            v,
            selected.lower_bounds,
            selected.valid,
            sm_scale=q.shape[-1] ** -0.5,
            kernel_config=config,
            q_offset=jnp.zeros((1,), dtype=jnp.int32),
        )

    output, pullback = jax.vjp(fused, selected.q, selected.k, selected.v)
    dq, dk, dv = pullback(selected.sparse_dout)
    positions = selected.positions
    output_error = jnp.max(jnp.abs(output.astype(jnp.float32) - selected.original_output.astype(jnp.float32)))
    return AttentionReplay(
        selected.q[:, positions],
        selected.k,
        selected.v,
        selected.sparse_dout[:, positions],
        output[:, positions],
        dq[:, positions],
        dk,
        dv,
        positions[None, :],
        selected.lower_bounds[:, positions],
        selected.valid[:, positions],
        selected.kv_group[None],
        selected.local_row[None],
        output_error[None],
        selected.selection_fallback_count[None],
    )


@eqx.filter_jit
def replay_attention_panel(capture: AttentionCapture, *, layer_index: int, query_count: int = 4) -> AttentionReplay:
    """Replay one native fused sparse VJP per rank, returning leading-rank arrays."""
    array_spec = P(BATCH_AXES, None, None, None)
    metadata_spec = P(BATCH_AXES, None)
    specs = AttentionCapture(
        array_spec, array_spec, array_spec, array_spec, array_spec, metadata_spec, metadata_spec, P(), P()
    )
    values = jax.tree.map(jax.sharding.reshard, capture, specs)
    return jax.shard_map(
        lambda local: _replay_local(local, layer_index=layer_index, query_count=query_count),
        in_specs=(specs,),
        out_specs=P(BATCH_AXES),
        check_vma=False,
    )(values)
