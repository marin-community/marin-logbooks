# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-head sparse native attention mitigation replay inside explicit shards."""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from levanter.grug.attention._fa4_cute_backend import (
    _import_cutlass_cute,
    _native_backward_gradients,
    segmented_flash_attention_backward,
    segmented_flash_attention_forward,
)
from levanter.grug.attention._fa4_cute_config import sm100_flash4_cute_kernel_config

from experiments.grug.moe_hero_ep.attention_mitigation_accumulators import decode_native_accumulator
from experiments.grug.moe_hero_ep.attention_mitigation_native import native_operand_replay
from experiments.grug.moe_hero_ep.attention_mitigation_projection import project_native_attention_operands
from experiments.grug.moe_hero_ep.attention_mitigation_saved_output import saved_output_backward, saved_output_forward
from experiments.grug.moe_hero_ep.attention_softmax_capture import AttentionCapture
from experiments.grug.moe_hero_ep.attention_softmax_replay import BATCH_AXES, select_attention_panel

_BASE_REQUIRED_ZERO_GATE_NAMES = (
    "capture_tail_max_abs_error",
    "baseline_forward_max_abs_error",
    "decoder_q_final_cast_max_abs_error",
    "decoder_k_final_cast_max_abs_error",
    "decoder_q_scale_one_max_abs_error",
    "decoder_k_scale_one_max_abs_error",
)
EXPORT_CHECK_NAMES = (
    "export_invalid_probability_row_count",
    "export_nonfinite_score_count",
    "export_masked_nonzero_count",
    "export_q_contraction_rounding_bound_excess",
    "export_k_contraction_rounding_bound_excess",
    "export_q_contraction_max_abs_difference",
    "export_k_contraction_max_abs_difference",
)
REQUIRED_ZERO_GATE_NAMES = (*_BASE_REQUIRED_ZERO_GATE_NAMES, *EXPORT_CHECK_NAMES[:5])
REPLAY_GATE_NAMES = (
    *_BASE_REQUIRED_ZERO_GATE_NAMES,
    "fp32_output_cast_max_abs_error",
    "fp32_output_lse_max_abs_error",
    "instrumented_q_max_abs_difference",
    "instrumented_k_max_abs_difference",
    "instrumented_v_max_abs_difference",
    *EXPORT_CHECK_NAMES,
)


class MitigationVariant(NamedTuple):
    output: jax.Array
    dq: jax.Array
    dk: jax.Array
    dv: jax.Array


class MitigationReplay(NamedTuple):
    q: jax.Array
    k: jax.Array
    v: jax.Array
    dout: jax.Array
    positions: jax.Array
    lower_bounds: jax.Array
    valid: jax.Array
    local_row: jax.Array
    selection_fallback_count: jax.Array
    variants: dict[str, MitigationVariant]
    gates: jax.Array
    row_mass: jax.Array
    probability_mass: jax.Array
    correction: jax.Array
    post_projection_mass: jax.Array


class FullHeadSelection(NamedTuple):
    q: jax.Array
    k: jax.Array
    v: jax.Array
    dout: jax.Array
    original_output: jax.Array
    positions: jax.Array
    lower_bounds: jax.Array
    valid: jax.Array
    local_row: jax.Array
    selection_fallback_count: jax.Array


def select_full_head_panel(
    capture: AttentionCapture, *, rank: jax.Array, layer_index: int, query_count: int
) -> FullHeadSelection:
    """Keep the previous sequence/query selection but retain every Q/K/V head."""
    selected = select_attention_panel(capture, rank=rank, layer_index=layer_index, query_count=query_count)
    row = selected.local_row

    def sequence(value):
        return jax.lax.dynamic_slice_in_dim(value, row, 1, axis=0)

    dout = sequence(capture.dout)
    sparse_dout = jnp.zeros_like(dout).at[:, selected.positions].set(dout[:, selected.positions])
    return FullHeadSelection(
        sequence(capture.q),
        sequence(capture.k),
        sequence(capture.v),
        sparse_dout,
        sequence(capture.output),
        selected.positions,
        sequence(capture.lower_bounds),
        sequence(capture.valid),
        row,
        selected.selection_fallback_count,
    )


def _max_abs_difference(first: jax.Array, second: jax.Array) -> jax.Array:
    return jnp.max(jnp.abs(first.astype(jnp.float32) - second.astype(jnp.float32)))


def native_operand_export_checks(
    q: jax.Array,
    k: jax.Array,
    probability: jax.Array,
    score: jax.Array,
    decoded_q: jax.Array,
    decoded_k: jax.Array,
    positions: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    full_query_length: int,
) -> dict[str, jax.Array]:
    """Check selected native operand coverage, masking, and contractions.

    Inputs are sampled BQHD Q, full BKHD K, BQHK BF16 operands, decoded
    unscaled FP32 accumulators, and sampled BQ positions/bounds/validity.
    BF16 products are exactly representable in FP32 before underflow. For
    each independently rounded FP32 sum use gamma_n = n*u/(1-n*u), u=2^-24.
    Four gamma_n times sum(abs(products)) conservatively covers both sums,
    a rounded bound reduction, and tiled/atomic reduction order. Native
    padding is covered by full sequence length times GQA for dK. A 2*n*tiny
    floor covers each sum flushing FP32 subnormal products/partial sums.
    This verifies correspondence rather than requiring bitwise reduction order.
    """
    batch, queries, heads, width = q.shape
    kv_heads = k.shape[2]
    groups = heads // kv_heads
    grouped_score = score.reshape(batch, queries, kv_heads, groups, k.shape[1])
    grouped_q = q.reshape(batch, queries, kv_heads, groups, width)
    reproduced_q = jnp.einsum("bqhrk,bkhd->bqhrd", grouped_score, k, preferred_element_type=jnp.float32).reshape(q.shape)
    reproduced_k = jnp.einsum("bqhrk,bqhrd->bkhd", grouped_score, grouped_q, preferred_element_type=jnp.float32)
    absolute_q = jnp.einsum(
        "bqhrk,bkhd->bqhrd", jnp.abs(grouped_score), jnp.abs(k), preferred_element_type=jnp.float32
    ).reshape(q.shape)
    absolute_k = jnp.einsum(
        "bqhrk,bqhrd->bkhd", jnp.abs(grouped_score), jnp.abs(grouped_q), preferred_element_type=jnp.float32
    )

    def rounding_excess(actual, reproduced, absolute, count):
        unit_roundoff = 2**-24
        gamma = (count * unit_roundoff) / (1 - count * unit_roundoff)
        bound = jnp.float32(4 * gamma) * absolute + jnp.float32(2 * count * jnp.finfo(jnp.float32).tiny)
        delta = jnp.abs(actual - reproduced)
        excess = jnp.maximum(delta - bound, 0)
        return jnp.max(excess), jnp.max(delta)

    q_excess, q_delta = rounding_excess(decoded_q, reproduced_q, absolute_q, k.shape[1])
    k_excess, k_delta = rounding_excess(decoded_k, reproduced_k, absolute_k, full_query_length * groups)
    key_positions = jnp.arange(k.shape[1])[None, None, :]
    allowed = valid[:, :, None] & (key_positions >= lower_bounds[:, :, None])
    allowed = allowed & (key_positions <= positions[:, :, None])
    masked_nonzero = (~allowed[:, :, None, :]) & ((probability != 0) | (score != 0))
    probability_mass = jnp.sum(probability.astype(jnp.float32), axis=-1)
    invalid_probability = valid[:, :, None] & (
        (probability_mass <= 0) | ~jnp.isfinite(probability_mass) | jnp.any(probability < 0, axis=-1)
    )
    values = (
        jnp.sum(invalid_probability, dtype=jnp.float32),
        jnp.sum(~jnp.isfinite(score), dtype=jnp.float32),
        jnp.sum(masked_nonzero, dtype=jnp.float32),
        q_excess,
        k_excess,
        q_delta,
        k_delta,
    )
    return dict(zip(EXPORT_CHECK_NAMES, values, strict=True))


def replay_mitigation_local(
    capture: AttentionCapture, *, rank: jax.Array, layer_index: int, query_count: int
) -> MitigationReplay:
    """Compare native variants on one local sequence with sparse real dO.

    Calls native kernels only when executed. Optional GPU dependencies are
    resolved by the underlying launchers, never at module import on CPU.
    The FP32-shadow variant returns baseline O and uses baseline LSE, so it
    changes only the D reduction supplied to backward. The optimized FP32-O
    variant has its own forward and LSE, whose differences remain measured.
    """
    selected = select_full_head_panel(capture, rank=rank, layer_index=layer_index, query_count=query_count)
    q, k, v, dout = selected.q, selected.k, selected.v, selected.dout
    if q.shape[-1] != 128 or q.shape[2] // k.shape[2] != 4:
        raise ValueError("Full-head mitigation replay requires native BF16 D128 GQA4")
    config = sm100_flash4_cute_kernel_config()
    scale = float(q.shape[-1] ** -0.5)
    offset = jnp.zeros((1,), dtype=jnp.int32)
    forward_args = dict(softmax_scale=scale, kernel_config=config, q_offset=offset)
    baseline_output, baseline_lse = segmented_flash_attention_forward(
        q, k, v, selected.lower_bounds, selected.valid, **forward_args
    )
    baseline = segmented_flash_attention_backward(
        q, k, v, baseline_output, dout, baseline_lse, selected.lower_bounds, selected.valid, **forward_args
    )
    fp32_output, fp32_state = saved_output_forward(
        q, k, v, selected.lower_bounds, selected.valid, sm_scale=scale, kernel_config=config, q_offset=offset
    )
    saved_args = dict(sm_scale=scale, kernel_config=config)
    fp32_gradients = saved_output_backward(fp32_state, dout, **saved_args)
    shadow_state = fp32_state._replace(lse=baseline_lse)
    shadow_gradients = saved_output_backward(shadow_state, dout, **saved_args)
    postcast_state = shadow_state._replace(output_fp32=baseline_output.astype(jnp.float32))
    postcast_gradients = saved_output_backward(postcast_state, dout, **saved_args)
    positions = selected.positions
    instrumented = native_operand_replay(
        q,
        k,
        v,
        baseline_output,
        dout,
        baseline_lse,
        selected.lower_bounds,
        selected.valid,
        positions[None],
        **saved_args,
    )
    tile_m, tile_n = config.sm100_backward.tile
    decoded_q = decode_native_accumulator(instrumented.accumulators[0], tensor_shape=q.shape, tile_m=tile_m)
    decoded_k = decode_native_accumulator(instrumented.accumulators[1], tensor_shape=k.shape, tile_m=tile_n)
    scale_one_gradients = _native_backward_gradients(
        _import_cutlass_cute(),
        (q, k, v),
        instrumented.accumulators,
        tile_rows=(tile_m, tile_n, tile_n),
        arch=100,
        num_threads=config.sm100_backward.postprocess_threads,
        softmax_scale=1.0,
    )
    projection = project_native_attention_operands(
        q[:, positions],
        k,
        instrumented.probability_operand,
        instrumented.score_gradient_operand,
        decoded_q[:, positions],
        decoded_k,
        softmax_scale=scale,
    )
    export_checks = native_operand_export_checks(
        q[:, positions],
        k,
        instrumented.probability_operand,
        instrumented.score_gradient_operand,
        decoded_q[:, positions],
        decoded_k,
        positions[None],
        selected.lower_bounds[:, positions],
        selected.valid[:, positions],
        full_query_length=q.shape[1],
    )

    def variant(output, gradients):
        dq, dk, dv = gradients
        return MitigationVariant(output[:, positions], dq[:, positions], dk, dv)

    variants = {
        "baseline": variant(baseline_output, baseline),
        "fp32_saved_output": variant(fp32_output, fp32_gradients),
        "fp32_saved_output_shadow": variant(baseline_output, shadow_gradients),
        "postcast_output_control": variant(baseline_output, postcast_gradients),
        "native_instrumented": variant(baseline_output, instrumented.gradients),
        "native_gproj_hybrid": MitigationVariant(
            baseline_output[:, positions], projection.dq, projection.dk, instrumented.gradients[2]
        ),
    }
    finite_lse_difference = jnp.where(
        jnp.isneginf(fp32_state.lse) & jnp.isneginf(baseline_lse), 0, fp32_state.lse - baseline_lse
    )
    gates = jnp.stack(
        (
            capture.tail_output_max_abs_error,
            _max_abs_difference(baseline_output, selected.original_output),
            _max_abs_difference((decoded_q * scale).astype(q.dtype), instrumented.gradients[0]),
            _max_abs_difference((decoded_k * scale).astype(k.dtype), instrumented.gradients[1]),
            _max_abs_difference(decoded_q.astype(q.dtype), scale_one_gradients[0]),
            _max_abs_difference(decoded_k.astype(k.dtype), scale_one_gradients[1]),
            _max_abs_difference(fp32_output, baseline_output),
            jnp.max(jnp.abs(finite_lse_difference)),
            *(
                _max_abs_difference(actual, expected)
                for actual, expected in zip(instrumented.gradients, baseline, strict=True)
            ),
            *(export_checks[name] for name in EXPORT_CHECK_NAMES),
        )
    )
    return MitigationReplay(
        q[:, positions],
        k,
        v,
        dout[:, positions],
        positions[None],
        selected.lower_bounds[:, positions],
        selected.valid[:, positions],
        selected.local_row[None],
        selected.selection_fallback_count[None],
        variants,
        gates[None],
        projection.row_mass,
        projection.probability_mass,
        projection.correction,
        projection.post_projection_mass,
    )


@eqx.filter_jit
def replay_attention_mitigations(
    capture: AttentionCapture, *, layer_index: int, query_count: int = 4
) -> MitigationReplay:
    """Replay native variants on every rank with explicit local head layouts."""
    array_spec = P(BATCH_AXES, None, None, None)
    metadata_spec = P(BATCH_AXES, None)
    specs = AttentionCapture(
        array_spec, array_spec, array_spec, array_spec, array_spec, metadata_spec, metadata_spec, P(), P()
    )
    values = jax.tree.map(jax.sharding.reshard, capture, specs)
    return jax.shard_map(
        lambda local: replay_mitigation_local(
            local, rank=jax.lax.axis_index(BATCH_AXES), layer_index=layer_index, query_count=query_count
        ),
        in_specs=(specs,),
        out_specs=P(BATCH_AXES),
        check_vma=False,
    )(values)


def smoke_local_replay(*, sequence_length: int, query_heads: int) -> MitigationReplay:
    """Execute the complete native replay on a small deterministic packed case.

    Intended GPU smoke inputs are (128,4) and (256,8). This has no CPU fallback:
    success requires compiling and running the native kernels and decoder.
    """
    if (sequence_length, query_heads) not in ((128, 4), (256, 8)):
        raise ValueError("Use a declared native smoke shape: (128,4) or (256,8)")
    shape = (1, sequence_length, query_heads, 128)
    key_shape = (1, sequence_length, query_heads // 4, 128)
    positions = jnp.arange(sequence_length)
    if sequence_length == 128:
        selected = jnp.asarray([31, 63, 95, 127])
        q = jnp.zeros(shape, jnp.bfloat16).at[:, selected, :, 0].set(jnp.bfloat16(10 * 128**0.5))
        k = jnp.zeros(key_shape, jnp.bfloat16).at[:, selected, :, 0].set(1)
        v = jnp.zeros(key_shape, jnp.bfloat16).at[:, selected, :, 0].set(1)
        dout = jnp.zeros(shape, jnp.bfloat16).at[:, selected, :, 0].set(1)
        lower = jnp.maximum(positions - 1, 0)[None]
        valid = jnp.zeros((1, sequence_length), dtype=jnp.bool_).at[:, selected].set(True)
    else:
        q = (jax.random.normal(jax.random.PRNGKey(701), shape) * 0.5).astype(jnp.bfloat16)
        k = (jax.random.normal(jax.random.PRNGKey(702), key_shape) * 0.5).astype(jnp.bfloat16)
        v = jax.random.normal(jax.random.PRNGKey(703), key_shape).astype(jnp.bfloat16)
        dout = (jax.random.normal(jax.random.PRNGKey(704), shape) * 0.25).astype(jnp.bfloat16)
        segment_length = sequence_length // 2
        lower = jnp.maximum((positions // segment_length) * segment_length, positions - 47)[None]
        valid = jnp.ones((1, sequence_length), dtype=jnp.bool_).at[:, 0].set(False)
    output, _ = segmented_flash_attention_forward(
        q,
        k,
        v,
        lower,
        valid,
        softmax_scale=128**-0.5,
        kernel_config=sm100_flash4_cute_kernel_config(),
        q_offset=jnp.zeros((1,), jnp.int32),
    )
    capture = AttentionCapture(q, k, v, output, dout, lower, valid, jnp.asarray(0.0), jnp.asarray(1.0))
    return replay_mitigation_local(capture, rank=jnp.asarray(0), layer_index=0, query_count=4)
