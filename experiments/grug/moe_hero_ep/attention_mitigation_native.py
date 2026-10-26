# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sparse native-operand GProj experiment, isolated from production attention.

The native SM100 mainloop exports its actual BF16 P/dS operands at selected
queries. Corrections use those operands and decoded FP32 accumulators, before
the final gradient casts. This hybrid is an accuracy experiment; exporting
selected rows does not establish the cost of a fully fused dense GProj kernel.
Every array is fully local inside the caller's explicit shard_map.
"""

import importlib
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call
from levanter.grug.attention._fa4_cute_backend import (
    _import_cutlass_cute,
    _native_backward_accum_output_shapes,
    _native_backward_accum_specs,
    _native_backward_gradients,
    _native_backward_preprocess,
    _packed_segment_backward_block_sparse_indices_with_full,
)
from levanter.grug.attention._fa4_cute_config import Flash4CuteKernelConfig
from levanter.grug.attention._fa4_cute_kernels import (
    _float32_zero_fill,
    _import_cute_dependencies,
    _native_broadcast_heads,
    _native_gmem_tensor_view,
    _patch_jax_array_list_tvm_ffi_converter,
)

from experiments.grug.moe_hero_ep.attention_mitigation_native_operands import native_sm100_operand_capture_module


class NativeOperandReplay(NamedTuple):
    accumulators: tuple[jax.Array, jax.Array, jax.Array]
    gradients: tuple[jax.Array, jax.Array, jax.Array]
    probability_operand: jax.Array
    score_gradient_operand: jax.Array


@cute_launcher_factory
def native_operand_launcher(modules: Any, *, config: Any, query_heads: int, kv_heads: int) -> Any:
    """Create the pinned instrumented mainloop without replacing native modules."""
    deps = _import_cute_dependencies(modules)
    cutlass, cute, cuda = deps.cutlass, deps.cute, deps.cuda
    native = native_sm100_operand_capture_module()
    sparsity = importlib.import_module("flash_attn.cute.block_sparsity")
    utils = importlib.import_module("flash_attn.cute.utils")
    _patch_jax_array_list_tvm_ffi_converter()
    broadcast_heads = _native_broadcast_heads(modules)
    as_gmem = _native_gmem_tensor_view(modules)
    zero_fill = _float32_zero_fill(modules, num_threads=config.zero_fill_threads)

    @cute.jit
    def packed_mask(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        del head_idx, seqlen_info
        batch_idx = utils.ssa_to_scalar(batch_idx)
        q_idx = utils.ssa_to_scalar(q_idx)
        kv_idx = utils.ssa_to_scalar(kv_idx)
        lower_bounds, valid, q_offset = aux_tensors[0], aux_tensors[1], aux_tensors[2]
        in_bounds = cute.elem_less(q_idx, lower_bounds.shape[1])
        safe_q = q_idx if in_bounds else lower_bounds.shape[1] - 1
        allowed = (
            in_bounds
            and valid[batch_idx, safe_q] != 0
            and cute.elem_less(lower_bounds[batch_idx, safe_q], kv_idx + 1)
            and cute.elem_less(kv_idx, q_idx + q_offset[0] + 1)
        )
        return utils.scalar_to_ssa(allowed, cutlass.Boolean)

    class ZeroPanel:
        @cute.jit
        def __call__(self, tensor: cute.Tensor, stream: cuda.CUstream):
            self.kernel(tensor).launch(
                grid=[cute.ceil_div(cute.size(tensor), 256), 1, 1], block=[256, 1, 1], stream=stream
            )

        @cute.kernel
        def kernel(self, tensor: cute.Tensor):
            thread, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            flat = cute.make_tensor(tensor.iterator, cute.make_layout(cute.size(tensor)))
            index = cutlass.Int64(block) * 256 + cutlass.Int64(thread)
            if index < cute.size(flat):
                flat[index] = cutlass.BFloat16(0.0)

    zero_panel = ZeroPanel()
    backward = native.FlashAttentionBackwardSm100(
        128,
        128,
        is_causal=False,
        is_local=False,
        qhead_per_kvhead=query_heads // kv_heads,
        tile_m=config.tile[0],
        tile_n=config.tile[1],
        cluster_size=1,
        use_2cta_instrs=False,
        deterministic=False,
        mask_mod=packed_mask,
        has_aux_tensors=True,
        kv_subtile_factor=1,
    )

    @cute.jit
    def launch(
        stream: cuda.CUstream,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        dout: cute.Tensor,
        lse_log2: cute.Tensor,
        dpsum: cute.Tensor,
        lower_bounds: cute.Tensor,
        valid: cute.Tensor,
        q_offset: cute.Tensor,
        partial_count: cute.Tensor,
        partial_index: cute.Tensor,
        full_count: cute.Tensor,
        full_index: cute.Tensor,
        positions: cute.Tensor,
        dq_accum: cute.Tensor,
        dk_accum: cute.Tensor,
        dv_accum: cute.Tensor,
        probability: cute.Tensor,
        score_gradient: cute.Tensor,
        *,
        softmax_scale: cutlass.Float32,
    ):
        blocks = sparsity.BlockSparseTensors(
            broadcast_heads(partial_count, query_heads),
            broadcast_heads(partial_index, query_heads),
            broadcast_heads(full_count, query_heads),
            broadcast_heads(full_index, query_heads),
        )
        zero_fill(dq_accum, stream)
        zero_fill(dk_accum, stream)
        zero_fill(dv_accum, stream)
        zero_panel(probability, stream)
        zero_panel(score_gradient, stream)
        backward(
            q,
            k,
            v,
            dout,
            as_gmem(lse_log2),
            as_gmem(dpsum),
            as_gmem(dq_accum),
            as_gmem(dk_accum),
            as_gmem(dv_accum),
            softmax_scale,
            aux_data=utils.AuxData(tensors=(lower_bounds, valid, q_offset, positions, probability, score_gradient)),
            blocksparse_tensors=blocks,
            stream=stream,
        )

    return launch


def native_operand_replay(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    output: jax.Array,
    dout: jax.Array,
    lse: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    positions: jax.Array,
    *,
    sm_scale: float,
    kernel_config: Flash4CuteKernelConfig,
) -> NativeOperandReplay:
    """Run actual native sparse VJP and export its selected post-cast operands."""
    config = kernel_config.sm100_backward
    if config is None or q.dtype != jnp.bfloat16 or q.shape[-1] != 128:
        raise ValueError("Operand instrumentation requires native SM100 BF16 D128")
    modules = _import_cutlass_cute()
    offset = jnp.zeros((1,), jnp.int32)
    sparse = _packed_segment_backward_block_sparse_indices_with_full(
        lower_bounds,
        valid,
        key_sequence_length=k.shape[1],
        q_offset=offset,
        tile_m=config.tile[0],
        tile_n=config.tile[1],
    )
    dpsum, lse_log2 = _native_backward_preprocess(
        modules, q, output, dout, lse, tile=config.tile, softmax_scale=sm_scale
    )
    inputs, outputs = _native_backward_accum_specs(modules, vector_elems=8)
    panel_shape = (q.shape[0], positions.shape[1], q.shape[2], k.shape[1])
    scalar_spec = modules.cjax.TensorSpec(static=True)
    call = cutlass_call(
        native_operand_launcher(modules, config=config, query_heads=q.shape[2], kv_heads=k.shape[2]),
        output_shape_dtype=(
            *_native_backward_accum_output_shapes(q, k, v, config.tile),
            jax.ShapeDtypeStruct(panel_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(panel_shape, jnp.bfloat16),
        ),
        input_spec=(*inputs, scalar_spec),
        output_spec=(*outputs, scalar_spec, scalar_spec),
        use_static_tensors=True,
        softmax_scale=sm_scale,
    )
    dq, dk, dv, probability, score_gradient = call(
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        lower_bounds,
        valid.astype(jnp.int32),
        offset,
        sparse.partial_block_cnt,
        sparse.partial_block_idx,
        sparse.full_block_cnt,
        sparse.full_block_idx,
        positions,
    )
    accumulators = (dq, dk, dv)
    gradients = _native_backward_gradients(
        modules,
        (q, k, v),
        accumulators,
        tile_rows=(config.tile[0], config.tile[1], config.tile[1]),
        arch=100,
        num_threads=config.postprocess_threads,
        softmax_scale=sm_scale,
    )
    return NativeOperandReplay(accumulators, gradients, probability, score_gradient)
