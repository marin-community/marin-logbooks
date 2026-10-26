# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Post-cast GProj operators for native-hybrid and reconstructed prototypes.

Implements Algorithm 1 of https://arxiv.org/html/2609.34272v1 on explicitly
materialized BF16 probability and score-gradient operands. The actual consumed
BF16 probability mass, rather than an assumed unit mass, normalizes the
post-cast correction. Q and K both receive the correction; the K correction
uses BF16(lambda * Q), as in the paper's second pass.

``project_native_attention_operands`` consumes genuine exported native operands
and native accumulators. It supplies the correction stage of an instrumented
native+XLA hybrid, not a fully fused kernel port. Its O(B*Q*H*K) export storage
is suitable for sampled queries, not a production fallback.

``projected_attention_backward`` instead reconstructs scores/probabilities and
their reduction order. Even with native saved LSE it does not inspect native
backward operands. Results and timing for that separate standalone prototype
must not be attributed to the native-hybrid intervention. The paper's FA3 cost
and accuracy results do not transfer to either FA4 prototype.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class ProjectionBackward(NamedTuple):
    dq: jax.Array
    dk: jax.Array
    dv: jax.Array
    unprojected_dq: jax.Array
    unprojected_dk: jax.Array
    row_mass: jax.Array
    probability_mass: jax.Array
    correction: jax.Array
    post_projection_mass: jax.Array


class ProjectedQK(NamedTuple):
    dq: jax.Array
    dk: jax.Array
    row_mass: jax.Array
    probability_mass: jax.Array
    correction: jax.Array
    post_projection_mass: jax.Array


@dataclass(frozen=True)
class ProjectionReference:
    """Exact-arithmetic projection of already quantized BF16 operands."""

    dq: np.ndarray
    dk: np.ndarray
    projected_scores: np.ndarray
    row_mass: np.ndarray
    probability_mass: np.ndarray
    correction: np.ndarray


def projection_reference(
    q: np.ndarray,
    k: np.ndarray,
    score_operand: np.ndarray,
    probability_operand: np.ndarray,
    key_mask: np.ndarray,
    *,
    softmax_scale: float,
) -> ProjectionReference:
    """Apply the mathematical post-cast projection and both contractions.

    q/k use BQHD/BKHD; operands use BQHK and must already be rounded to the
    desired operand dtype. FP64 arithmetic here isolates the projection from
    the extra BF16(lambda*Q) rounding of Algorithm 1's implemented key pass.
    """
    q, k, score, probability = [
        np.asarray(array, dtype=np.float64) for array in (q, k, score_operand, probability_operand)
    ]
    mask = np.asarray(key_mask, dtype=bool)[:, :, None, :]
    score = np.where(mask, score, 0)
    probability = np.where(mask, probability, 0)
    row_mass = score.sum(axis=-1)
    probability_mass = probability.sum(axis=-1)
    correction = np.divide(row_mass, probability_mass, out=np.zeros_like(row_mass), where=probability_mass != 0)
    projected = score - correction[..., None] * probability
    batch, query_count, heads, width = q.shape
    kv_heads = k.shape[2]
    groups = heads // kv_heads
    if heads % kv_heads or score.shape != (batch, query_count, heads, k.shape[1]):
        raise ValueError("Projection operands must match Q/K with contiguous GQA heads")
    grouped = projected.reshape(batch, query_count, kv_heads, groups, k.shape[1])
    dq = softmax_scale * np.einsum("bqhrk,bkhd->bqhrd", grouped, k)
    dk = softmax_scale * np.einsum("bqhrk,bqhrd->bkhd", grouped, q.reshape(batch, query_count, kv_heads, groups, width))
    return ProjectionReference(dq.reshape(q.shape), dk, projected, row_mass, probability_mass, correction)


def project_native_attention_operands(
    q: jax.Array,
    k: jax.Array,
    probability_operand: jax.Array,
    score_operand: jax.Array,
    dq_accumulator: jax.Array,
    dk_accumulator: jax.Array,
    *,
    softmax_scale: float,
) -> ProjectedQK:
    """Correct genuine native post-cast operands and pre-final-cast Q/K sums.

    Q/K use BQHD/BKHD; operands use BQHK. Native FP32 accumulators must first
    be decoded into those logical layouts, retaining their UNSCALED units.
    Q and dQ may contain just the sampled rows. K and dK retain every key.
    The dK accumulator must contain only those sampled rows' cotangents.

    The instrumented native pass supplies the actual BF16 P and dS operands.
    This XLA correction stage is a hybrid prototype, not a fully fused port.
    It measures native row mass after the actual cast, corrects the FP32
    accumulators, applies softmax scale once, then rounds the final gradients.
    """
    batch, query_count, heads, width = q.shape
    kv_heads = k.shape[2]
    groups = heads // kv_heads
    shape = (batch, query_count, heads, k.shape[1])
    if any(array.dtype != jnp.bfloat16 for array in (q, k, probability_operand, score_operand)):
        raise ValueError("Native projection requires actual BF16 operands")
    if heads % kv_heads or probability_operand.shape != shape or score_operand.shape != shape:
        raise ValueError("Native projection operands must use matching BQHK and contiguous GQA heads")
    if dq_accumulator.shape != q.shape or dk_accumulator.shape != k.shape:
        raise ValueError("Decode native accumulators into matching logical Q/K layouts before projection")
    if dq_accumulator.dtype != jnp.float32 or dk_accumulator.dtype != jnp.float32:
        raise ValueError("Projection must precede native FP32 accumulator final casts")
    grouped_shape = (batch, query_count, kv_heads, groups, k.shape[1])
    probabilities = probability_operand.reshape(grouped_shape)
    row_mass = jnp.sum(score_operand.astype(jnp.float32), axis=-1, dtype=jnp.float32)
    probability_mass = jnp.sum(probability_operand.astype(jnp.float32), axis=-1, dtype=jnp.float32)
    safe_mass = jnp.where(probability_mass != 0, probability_mass, jnp.float32(1))
    correction = row_mass / safe_mass
    probability_keys = jnp.einsum("bqhrk,bkhd->bqhrd", probabilities, k, preferred_element_type=jnp.float32)
    query_correction = correction[..., None] * probability_keys.reshape(q.shape)
    corrected_query = (correction[..., None] * q.astype(jnp.float32)).astype(jnp.bfloat16)
    corrected_query = corrected_query.reshape(batch, query_count, kv_heads, groups, width)
    key_correction = jnp.einsum("bqhrk,bqhrd->bkhd", probabilities, corrected_query, preferred_element_type=jnp.float32)
    scale = jnp.float32(softmax_scale)
    return ProjectedQK(
        ((dq_accumulator - query_correction) * scale).astype(jnp.bfloat16),
        ((dk_accumulator - key_correction) * scale).astype(jnp.bfloat16),
        row_mass,
        probability_mass,
        correction,
        row_mass - correction * probability_mass,
    )


def projected_attention_backward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    dout: jax.Array,
    stored_output: jax.Array,
    key_mask: jax.Array,
    *,
    softmax_scale: float,
    log_normalizer: jax.Array | None = None,
) -> ProjectionBackward:
    """Run the standalone BF16-operand/FP32-accumulator GProj prototype.

    Inputs use BQHD/BKHD; mask uses BQK. Optional native saved LSE uses BQH
    and natural-log units. Q may be sampled and dK aggregates only those dO
    rows. Outputs use BF16, matching the native gradient boundary. Row
    statistics use BQH and describe the explicitly reconstructed operands.

    Saved O may be BF16 or genuine pre-cast FP32. This allows comparing the
    projection independently and together with the saved-output intervention.
    """
    batch, query_count, heads, width = q.shape
    kv_heads = k.shape[2]
    if heads % kv_heads or key_mask.shape != (batch, query_count, k.shape[1]):
        raise ValueError("Projection inputs require contiguous GQA heads and a BQK mask")
    if any(array.dtype != jnp.bfloat16 for array in (q, k, v, dout)):
        raise ValueError("The sampled-panel projection requires BF16 Q/K/V/dO operands")
    groups = heads // kv_heads
    q_grouped = q.reshape(batch, query_count, kv_heads, groups, width)
    dout_grouped = dout.reshape(batch, query_count, kv_heads, groups, v.shape[-1])
    scores = jnp.einsum("bqhrd,bkhd->bqhrk", q_grouped, k, preferred_element_type=jnp.float32)
    scores = scores * jnp.float32(softmax_scale)
    mask = key_mask[:, :, None, None, :]
    scores = jnp.where(mask, scores, -jnp.inf)
    if log_normalizer is None:
        lse = jax.nn.logsumexp(scores, axis=-1)
    else:
        lse = log_normalizer.reshape(batch, query_count, kv_heads, groups)
    # Fully masked rows must not evaluate exp(-inf - -inf).
    safe_lse = jnp.where(jnp.any(mask, axis=-1), lse, jnp.float32(0))
    probabilities = jnp.where(mask, jnp.exp(scores - safe_lse[..., None]), jnp.float32(0))
    probability_operand = probabilities.astype(jnp.bfloat16)
    dp = jnp.einsum("bqhrd,bkhd->bqhrk", dout_grouped, v, preferred_element_type=jnp.float32)
    saved_correction = jnp.sum(stored_output.astype(jnp.float32) * dout.astype(jnp.float32), axis=-1)
    saved_correction = saved_correction.reshape(batch, query_count, kv_heads, groups)
    score_operand = (probabilities * (dp - saved_correction[..., None])).astype(jnp.bfloat16)
    raw_dq = jnp.einsum("bqhrk,bkhd->bqhrd", score_operand, k, preferred_element_type=jnp.float32)
    raw_dk = jnp.einsum("bqhrk,bqhrd->bkhd", score_operand, q_grouped, preferred_element_type=jnp.float32)
    projected = project_native_attention_operands(
        q,
        k,
        probability_operand.reshape(batch, query_count, heads, k.shape[1]),
        score_operand.reshape(batch, query_count, heads, k.shape[1]),
        raw_dq.reshape(q.shape),
        raw_dk,
        softmax_scale=softmax_scale,
    )
    dv = jnp.einsum("bqhrk,bqhrd->bkhd", probability_operand, dout_grouped, preferred_element_type=jnp.float32)
    scale = jnp.float32(softmax_scale)
    return ProjectionBackward(
        projected.dq,
        projected.dk,
        dv.astype(jnp.bfloat16),
        (raw_dq * scale).reshape(q.shape).astype(jnp.bfloat16),
        (raw_dk * scale).astype(jnp.bfloat16),
        projected.row_mass,
        projected.probability_mass,
        projected.correction,
        projected.post_projection_mass,
    )
