# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experimental native SM100 attention retaining pre-BF16 FP32 output.

The pinned FlashAttention SM100 kernel accepts an independent output dtype.
Requesting Float32 preserves its normalized accumulator before the usual BF16
epilogue cast; converting the returned BF16 output back to FP32 would not do so.
This changes the native output layout and KV pipeline depth, so cast-output
parity and end-to-end cost must be measured against the untouched baseline.
The FP32 candidate uses one query stage at the original 128x128 MMA tile.
Two query stages exhaust the KV pipeline slots and deadlock on multiple K tiles.

All functions consume fully local BSHD arrays inside the caller's shard_map.
The production backend remains unchanged. This adapter reuses its pinned native
launcher, sparse metadata, backward mainloop, and gradient postprocessing.
"""

import math
from dataclasses import replace
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from levanter.cutlass_kernel_cache import cutlass_call
from levanter.grug.attention._fa4_cute_backend import (
    _cutlass_attention_forward_specs,
    _import_cutlass_cute,
    _native_backward_accum_output_shapes,
    _native_backward_accum_specs,
    _native_backward_gradients,
    _packed_segment_backward_block_sparse_indices_with_full,
    _packed_segment_forward_block_sparse_indices_with_full,
    _validate_forward_inputs,
    segmented_flash_attention_forward,
)
from levanter.grug.attention._fa4_cute_config import Flash4CuteKernelConfig
from levanter.grug.attention._fa4_cute_kernels import (
    segmented_flash_attention_backward_sm100_launcher,
    segmented_flash_attention_forward_sm100_launcher,
)

LOG2_E = math.log2(math.e)
FP32_FORWARD_Q_STAGE = 1
SM100_FORWARD_SHARED_MEMORY_BUDGET = 224 * 1024


class SavedOutputState(NamedTuple):
    q: jax.Array
    k: jax.Array
    v: jax.Array
    output_fp32: jax.Array
    lse: jax.Array
    lower_bounds: jax.Array
    valid: jax.Array
    q_offset: jax.Array


def saved_output_preprocess(
    output_fp32: jax.Array, dout: jax.Array, lse: jax.Array, *, tile_m: int
) -> tuple[jax.Array, jax.Array]:
    """Return padded native-layout D and LSElog2 using the genuine FP32 output.

    Unlike the baseline native preprocessor, this supports mixed FP32 O/BF16 dO.
    D uses an FP32 XLA reduction; its accumulation order is another measured
    candidate difference from the baseline native warp reduction.
    """
    if output_fp32.dtype != jnp.float32:
        raise ValueError("Saved attention output must be pre-cast float32")
    if output_fp32.shape != dout.shape or lse.shape != (
        output_fp32.shape[0],
        output_fp32.shape[2],
        output_fp32.shape[1],
    ):
        raise ValueError("O/dO/LSE shapes must match local BSHD/BHS layout")
    correction = jnp.sum(output_fp32 * dout.astype(jnp.float32), axis=-1, dtype=jnp.float32)
    correction = jnp.transpose(correction, (0, 2, 1))
    lse_log2 = jnp.where(jnp.isneginf(lse), jnp.float32(0), lse * jnp.float32(LOG2_E))
    padding = (-output_fp32.shape[1]) % tile_m
    pads = ((0, 0), (0, 0), (0, padding))
    return jnp.pad(correction, pads), jnp.pad(lse_log2, pads)


def saved_output_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    sm_scale: float,
    kernel_config: Flash4CuteKernelConfig,
    q_offset: jax.Array,
) -> tuple[jax.Array, SavedOutputState]:
    """Run native forward into FP32 saved state and return a public BF16 output."""
    _validate_forward_inputs(q, k, v, lower_bounds, valid, softmax_scale=sm_scale, q_offset=q_offset)
    if q.dtype != jnp.bfloat16 or q.shape[-1] != 128 or v.shape[-1] != 128:
        raise ValueError("Saved-output experiment requires native BF16 D128 attention")
    if q.shape[2] // k.shape[2] != 4:
        raise ValueError("Saved-output experiment requires Hero GQA4")
    config = kernel_config.sm100_forward
    if config is None or kernel_config.sm100_backward is None:
        raise ValueError("Saved-output experiment requires explicit SM100 forward/backward configs")
    if config.tile != (128, 128) or config.q_stage not in (1, 2):
        raise ValueError("FP32 saved-output candidate requires native 128x128 forward tiles")
    # Two query stages plus FP32 O leave only one aliased KV slot. The native
    # multi-key loop waits for next K before releasing V from the second query
    # stage, which deadlocks with one slot. One query stage retains four slots.
    config = replace(config, q_stage=FP32_FORWARD_Q_STAGE)
    query_and_output_bytes = config.q_stage * config.tile[0] * 128 * (2 + 4)
    kv_tile_bytes = config.tile[1] * 128 * 2
    kv_stages = (SM100_FORWARD_SHARED_MEMORY_BUDGET - query_and_output_bytes) // kv_tile_bytes
    if kv_stages < 2:
        raise ValueError("FP32 saved-output forward requires at least two native KV pipeline slots")
    modules = _import_cutlass_cute()
    sparse = _packed_segment_forward_block_sparse_indices_with_full(
        lower_bounds,
        valid,
        key_sequence_length=k.shape[1],
        q_offset=q_offset,
        tile_m=config.tile[0] * config.q_stage,
        tile_n=config.tile[1],
    )
    launcher = segmented_flash_attention_forward_sm100_launcher(
        modules,
        head_dim=q.shape[-1],
        head_dim_v=v.shape[-1],
        qhead_per_kvhead=q.shape[2] // k.shape[2],
        config=config,
    )
    inputs, outputs = _cutlass_attention_forward_specs(modules, vector_elems=8)
    metadata_spec = modules.cjax.TensorSpec(static=True)
    call = cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((*q.shape[:3], v.shape[-1]), jnp.float32),
            jax.ShapeDtypeStruct((q.shape[0], q.shape[2], q.shape[1]), jnp.float32),
        ),
        input_spec=(*inputs, metadata_spec, metadata_spec, metadata_spec, metadata_spec),
        output_spec=outputs,
        use_static_tensors=True,
        softmax_scale=sm_scale,
    )
    output_fp32, lse = call(
        q,
        k,
        v,
        lower_bounds,
        valid.astype(jnp.int32),
        q_offset,
        sparse.partial_block_cnt,
        sparse.partial_block_idx,
        sparse.full_block_cnt,
        sparse.full_block_idx,
    )
    state = SavedOutputState(q, k, v, output_fp32, lse, lower_bounds, valid, q_offset)
    return output_fp32.astype(q.dtype), state


def saved_output_backward(
    state: SavedOutputState,
    dout: jax.Array,
    *,
    sm_scale: float,
    kernel_config: Flash4CuteKernelConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run unchanged native backward with FP32-saved-output correction."""
    config = kernel_config.sm100_backward
    if config is None:
        raise ValueError("Saved-output backward requires native SM100 config")
    q, k, v = state.q, state.k, state.v
    modules = _import_cutlass_cute()
    sparse = _packed_segment_backward_block_sparse_indices_with_full(
        state.lower_bounds,
        state.valid,
        key_sequence_length=k.shape[1],
        q_offset=state.q_offset,
        tile_m=config.tile[0],
        tile_n=config.tile[1],
    )
    dpsum, lse_log2 = saved_output_preprocess(state.output_fp32, dout, state.lse, tile_m=config.tile[0])
    inputs, outputs = _native_backward_accum_specs(modules, vector_elems=8)
    backward = cutlass_call(
        segmented_flash_attention_backward_sm100_launcher(
            modules,
            head_dim=q.shape[-1],
            head_dim_v=v.shape[-1],
            qhead_per_kvhead=q.shape[2] // k.shape[2],
            config=config,
        ),
        output_shape_dtype=_native_backward_accum_output_shapes(q, k, v, config.tile),
        input_spec=inputs,
        output_spec=outputs,
        use_static_tensors=True,
        softmax_scale=sm_scale,
    )
    accumulators = backward(
        q,
        k,
        v,
        dout.astype(q.dtype),
        lse_log2,
        dpsum,
        state.lower_bounds,
        state.valid.astype(jnp.int32),
        state.q_offset,
        sparse.partial_block_cnt,
        sparse.partial_block_idx,
        sparse.full_block_cnt,
        sparse.full_block_idx,
    )
    return _native_backward_gradients(
        modules,
        (q, k, v),
        accumulators,
        tile_rows=(config.tile[0], config.tile[1], config.tile[1]),
        arch=100,
        num_threads=config.postprocess_threads,
        softmax_scale=sm_scale,
    )


@partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _saved_output_attention(q, k, v, lower_bounds, valid, q_offset, sm_scale, kernel_config):
    return saved_output_forward(
        q, k, v, lower_bounds, valid, sm_scale=sm_scale, kernel_config=kernel_config, q_offset=q_offset
    )[0]


def _saved_output_attention_fwd(q, k, v, lower_bounds, valid, q_offset, sm_scale, kernel_config):
    return saved_output_forward(
        q, k, v, lower_bounds, valid, sm_scale=sm_scale, kernel_config=kernel_config, q_offset=q_offset
    )


def _saved_output_attention_bwd(sm_scale, kernel_config, state, dout):
    dq, dk, dv = saved_output_backward(state, dout, sm_scale=sm_scale, kernel_config=kernel_config)
    return dq, dk, dv, None, None, None


_saved_output_attention.defvjp(_saved_output_attention_fwd, _saved_output_attention_bwd)


def saved_output_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    sm_scale: float,
    kernel_config: Flash4CuteKernelConfig,
    q_offset: jax.Array,
) -> jax.Array:
    """Return BF16 native attention with an FP32 pre-cast output VJP residual."""
    return _saved_output_attention(q, k, v, lower_bounds, valid, q_offset, sm_scale, kernel_config)


def shadow_saved_output_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    sm_scale: float,
    kernel_config: Flash4CuteKernelConfig,
    q_offset: jax.Array,
) -> tuple[jax.Array, SavedOutputState]:
    """Return untouched baseline output and separately computed FP32 shadow state.

    Both native forwards are intentional and must count toward candidate cost.
    Backward retains baseline LSE for probability reconstruction, using the
    genuine pre-cast shadow output only for its FP32 dO-dot-O correction.
    This explicit candidate never selects a fallback based on parity results.
    """
    baseline_output, baseline_lse = segmented_flash_attention_forward(
        q,
        k,
        v,
        lower_bounds,
        valid,
        softmax_scale=sm_scale,
        kernel_config=kernel_config,
        q_offset=q_offset,
    )
    _, shadow_state = saved_output_forward(
        q, k, v, lower_bounds, valid, sm_scale=sm_scale, kernel_config=kernel_config, q_offset=q_offset
    )
    return baseline_output, shadow_state._replace(lse=baseline_lse)


@partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _shadow_saved_output_attention(q, k, v, lower_bounds, valid, q_offset, sm_scale, kernel_config):
    return shadow_saved_output_forward(
        q, k, v, lower_bounds, valid, sm_scale=sm_scale, kernel_config=kernel_config, q_offset=q_offset
    )[0]


def _shadow_saved_output_attention_fwd(q, k, v, lower_bounds, valid, q_offset, sm_scale, kernel_config):
    return shadow_saved_output_forward(
        q, k, v, lower_bounds, valid, sm_scale=sm_scale, kernel_config=kernel_config, q_offset=q_offset
    )


_shadow_saved_output_attention.defvjp(_shadow_saved_output_attention_fwd, _saved_output_attention_bwd)


def shadow_saved_output_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    sm_scale: float,
    kernel_config: Flash4CuteKernelConfig,
    q_offset: jax.Array,
) -> jax.Array:
    """Return baseline BF16 attention with an explicit extra FP32-shadow VJP."""
    return _shadow_saved_output_attention(q, k, v, lower_bounds, valid, q_offset, sm_scale, kernel_config)
