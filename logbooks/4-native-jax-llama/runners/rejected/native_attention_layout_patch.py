"""Disposable reference-attention layout experiment; no repository source edit.

Transpose raw query/key/value arrays to the legacy grouped-head dimension order
before dot_general. Keep public BQHD/BKHD arrays, mask semantics and result dtype.
"""

import math

import jax
import jax.numpy as jnp

import levanter.grug.attention._core as attention_core


def aligned_reference_attention_math(q, k, v, mask, *, logits_dtype):
    head_dim = q.shape[-1]
    num_q_heads = q.shape[2]
    num_kv_heads = k.shape[2]
    if num_q_heads % num_kv_heads:
        raise ValueError(f"num_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
    groups = num_q_heads // num_kv_heads
    # Match legacy [batch, kv_heads, query_groups, query_position, head_dim].
    q = q.reshape(q.shape[0], q.shape[1], num_kv_heads, groups, head_dim).transpose(0, 2, 3, 1, 4)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    scale = 1.0 / math.sqrt(head_dim)
    scores = jax.lax.dot_general(
        q * scale,
        k,
        dimension_numbers=(((4,), (3,)), ((0, 1), (0, 1))),
    )

    explicit = None
    if mask is None:
        explicit = None
    elif isinstance(mask, attention_core.AttentionMask):
        explicit = mask.materialize_mask(scores.shape[-2], scores.shape[-1])
    else:
        explicit = mask

    if explicit is not None:
        if explicit.ndim == 2:
            explicit = explicit[None, :, :]
        if explicit.ndim != 3:
            raise ValueError(f"explicit mask must have shape [batch, q, k], got shape={explicit.shape}")
        if explicit.shape[0] not in (1, q.shape[0]):
            raise ValueError(f"explicit mask batch dim must be 1 or {q.shape[0]}, got {explicit.shape[0]}")
        if explicit.shape[1] != scores.shape[-2] or explicit.shape[2] != scores.shape[-1]:
            raise ValueError(
                "explicit mask must match attention shapes: "
                f"got mask={explicit.shape}, expected [batch,{scores.shape[-2]},{scores.shape[-1]}]"
            )

        explicit = explicit[:, None, None, :, :]
        if explicit.dtype == jnp.bool_:
            scores = jnp.where(explicit, scores, jnp.array(-1e9, dtype=scores.dtype))
        else:
            scores = scores + explicit
    if logits_dtype is not None:
        scores = scores.astype(logits_dtype)
    weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    context = jax.lax.dot_general(
        weights,
        v,
        dimension_numbers=(((4,), (2,)), ((0, 1), (0, 1))),
    )
    context = context.transpose(0, 3, 1, 2, 4)
    return context.reshape(context.shape[0], context.shape[1], num_q_heads, head_dim).astype(v.dtype)


attention_core._reference_attention_math = aligned_reference_attention_math
