# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen-weight Hero attention boundaries for checkpoint gradient diagnostics."""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from levanter.grug.attention import AttentionMask, align_kv_heads, attention, fa4_cute_segment_bounds

from experiments.grug.moe_hero_ep.model import (
    _BATCH_AXES,
    Block,
    CausalSelfAttention,
    _apply_rotary_embedding_fused,
    _partition_spec_of,
    _sequence_axis_of,
    rms_norm,
)


class AttentionInputs(NamedTuple):
    q: jax.Array
    k: jax.Array
    v: jax.Array


class AttentionCapture(NamedTuple):
    q: jax.Array
    k: jax.Array
    v: jax.Array
    output: jax.Array
    dout: jax.Array
    lower_bounds: jax.Array
    valid: jax.Array
    tail_output_max_abs_error: jax.Array
    tail_output_reference_rms: jax.Array


def attention_inputs(
    module: CausalSelfAttention, x: jax.Array, mask: AttentionMask, *, is_global: jax.Array
) -> AttentionInputs:
    """Prepare the recorded Hero's normalized, rotated, scaled attention operands."""
    cfg = module.cfg
    if not cfg.rope_fused:
        raise ValueError("This checkpoint diagnostic requires the Hero fused-RoPE recipe")
    mesh = get_abstract_mesh()
    if mesh is not None and not mesh.empty and int(mesh.shape.get("context", 1)) != 1:
        raise ValueError("The attention checkpoint diagnostic requires context axis size one")
    head_dim = cfg.inferred_head_dim
    q_flat = jnp.einsum("bsh,hd->bsd", x, module.w_q)
    k_flat = jnp.einsum("bsh,hd->bsd", x, module.w_k)
    v_flat = jnp.einsum("bsh,hd->bsd", x, module.w_v)
    segment_ids = mask.segment_ids[0] if mask.segment_ids is not None else None
    if module.sconv_k is not None:
        k_flat = module.sconv_k(k_flat, segment_ids)
    q = rearrange(q_flat, "... (n d) -> ... n d", d=head_dim)
    k = rearrange(k_flat, "... (n d) -> ... n d", d=head_dim)
    v = rearrange(v_flat, "... (n d) -> ... n d", d=head_dim)
    if cfg.local_kv_heads is not None and cfg.global_kv_heads is not None:
        kv_spec = P(_BATCH_AXES, _sequence_axis_of(x), None, None)

        def logical_kv(projection, num_kv_heads):
            projection = reshard(projection, kv_spec)
            if num_kv_heads == cfg.stored_kv_heads:
                return projection
            return align_kv_heads(projection[:, :, :num_kv_heads, :], num_q_heads=cfg.stored_kv_heads)

        k, v = jax.lax.cond(
            jnp.asarray(is_global, dtype=jnp.bool_),
            lambda kv: (logical_kv(kv[0], cfg.global_kv_heads), logical_kv(kv[1], cfg.global_kv_heads)),
            lambda kv: (logical_kv(kv[0], cfg.local_kv_heads), logical_kv(kv[1], cfg.local_kv_heads)),
            (k, v),
        )
    q, k = _apply_rotary_embedding_fused(
        rms_norm(q),
        rms_norm(k),
        seq_len=x.shape[1],
        head_dim=head_dim,
        rotary_dim=head_dim // 2,
        rope=cfg.rope,
        disable_rope=is_global,
    )
    q = q * cfg.qk_mult
    if cfg.qk_mult_long_scale != 1.0:
        q = q * jnp.where(is_global, cfg.qk_mult_long_scale, 1.0)
    return AttentionInputs(q, k, v)


def attention_finish(module: CausalSelfAttention, x: jax.Array, raw_output: jax.Array, v: jax.Array) -> jax.Array:
    """Apply the Hero's XSA, head gating, and output projection after fused attention."""
    aligned_v = align_kv_heads(v, num_q_heads=raw_output.shape[2])
    aligned_v = reshard(aligned_v, _partition_spec_of(raw_output) or P(_BATCH_AXES, None, None, "model"))
    dot = jnp.sum(raw_output * aligned_v, axis=-1, keepdims=True)
    v_norm_sq = jnp.sum(aligned_v * aligned_v, axis=-1, keepdims=True)
    result = raw_output - (dot / (v_norm_sq + 1e-6)) * aligned_v
    gate = 2 * jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, module.attn_gate))[..., None]
    result = gate * result
    seq_axis = _sequence_axis_of(x)
    result = jnp.reshape(
        result,
        (*result.shape[:-2], result.shape[-2] * result.shape[-1]),
        out_sharding=P(_BATCH_AXES, seq_axis, "model"),
    )
    return jnp.einsum("bsh,hd->bsd", result, module.w_o, out_sharding=P(_BATCH_AXES, seq_axis, None))


class _AttentionOutput(eqx.Module):
    module: CausalSelfAttention
    raw_output: jax.Array
    v: jax.Array

    def __call__(self, x, mask, disable_rope=False, is_global=False):
        return attention_finish(self.module, x, self.raw_output, self.v)


@eqx.filter_jit
def block_forward(block: Block, x: jax.Array, mask: AttentionMask, is_global: jax.Array) -> jax.Array:
    return block(x, mask, disable_rope=is_global, is_global=is_global)[0]


@eqx.filter_jit
def block_hidden_vjp(block: Block, x: jax.Array, dy: jax.Array, mask: AttentionMask, is_global: jax.Array) -> jax.Array:
    """Propagate an actual block cotangent through frozen parameters."""
    _, pullback = jax.vjp(lambda value: block(value, mask, disable_rope=is_global, is_global=is_global)[0], x)
    return pullback(dy)[0]


@eqx.filter_jit
def selected_attention_vjp(
    block: Block, x: jax.Array, dy: jax.Array, mask: AttentionMask, is_global: jax.Array
) -> AttentionCapture:
    """Capture real attention operands and the full downstream raw-attention cotangent.

    The original Block supplies its residual, branch convolution, MLP, and shared
    expert computations. Only its attention module is replaced with the unchanged
    post-attention computation evaluated from an independent raw output boundary.
    """
    attn_in = block.attn_gated_norm(block.rms_attn(x))
    q, k, v = attention_inputs(block.attn, attn_in, mask, is_global=is_global)
    output = attention(q, k, v, mask, implementation=block.attn.cfg.attention_implementation)

    def from_output(raw_output):
        replacement = _AttentionOutput(block.attn, raw_output, v)
        patched_block = eqx.tree_at(lambda value: value.attn, block, replacement)
        return patched_block(x, mask, disable_rope=is_global, is_global=is_global)[0]

    tail_output, pullback = jax.vjp(from_output, output)
    dout = pullback(dy)[0]
    reference_output = block(x, mask, disable_rope=is_global, is_global=is_global)[0]
    reference_f32 = reference_output.astype(jnp.float32)
    tail_output_max_abs_error = jnp.max(jnp.abs(tail_output.astype(jnp.float32) - reference_f32))
    tail_output_reference_rms = jnp.sqrt(jnp.mean(jnp.square(reference_f32)))
    if mask.fa4_bounds is not None:
        lower_bounds, valid = mask.fa4_bounds
    else:
        lower_bounds, valid = fa4_cute_segment_bounds(
            mask, batch_size=x.shape[0], seq_len=x.shape[1], sliding_window=mask.sliding_window
        )
    return AttentionCapture(
        q, k, v, output, dout, lower_bounds, valid, tail_output_max_abs_error, tail_output_reference_rms
    )
