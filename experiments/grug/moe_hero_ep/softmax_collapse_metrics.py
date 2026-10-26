# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Additive output-logit diagnostics, independent of the production fused CE kernel."""

import math
from itertools import pairwise

import jax
import jax.numpy as jnp

FP32_MARGIN_CLIFF = 24 * math.log(2)
MARGIN_HISTOGRAM_EDGES = (-math.inf, 0, 4, 8, 12, 14, 16, FP32_MARGIN_CLIFF, 18, 24, 40, math.inf)


def softmax_collapse_metrics(logits: jax.Array, targets: jax.Array, weights: jax.Array) -> dict[str, jax.Array]:
    """Summarize ``[tokens, vocab]`` logits using fp32 dense CE and stable margins.

    Every value is additive over token blocks, including the fixed-size histogram.
    Counts and sums use finite, positive-weight rows with finite logits and an
    in-range target. Zero-weight rows are masked; negative/nonfinite weights and
    other excluded rows have explicit counts. Token rates use ``valid_count``;
    weights are reported separately. Histogram intervals include their lower edge.

    The ``dense_fp32_*`` metrics use ordinary dense ``logsumexp``. They describe
    that calculation, not the production streaming kernel's reduction order.
    All gradients are per-token pure CE, before weighting or the LM-head z-loss.
    """
    logits = logits.astype(jnp.float32)
    finite_weight = jnp.isfinite(weights)
    positive_weight = finite_weight & (weights > 0)
    finite_logits = jnp.all(jnp.isfinite(logits), axis=-1)
    valid_target = (targets >= 0) & (targets < logits.shape[-1])
    valid = positive_weight & finite_logits & valid_target
    safe_logits = jnp.where(valid[:, None], logits, 0.0)
    safe_targets = jnp.clip(targets, 0, logits.shape[-1] - 1)
    target_logit = jnp.take_along_axis(safe_logits, safe_targets[:, None], axis=-1)[:, 0]
    is_target = jnp.arange(logits.shape[-1])[None, :] == safe_targets[:, None]
    shifted_others = jnp.where(is_target, -jnp.inf, safe_logits - target_logit[:, None])
    margin = -jax.nn.logsumexp(shifted_others, axis=-1)
    stable_ce = jax.nn.softplus(-margin)
    stable_target_gradient = -jax.nn.sigmoid(-margin)
    dense_lse = jax.nn.logsumexp(safe_logits, axis=-1)
    dense_ce = dense_lse - target_logit
    dense_target_gradient = jnp.exp(target_logit - dense_lse) - 1.0
    non_target_mass = jnp.sum(jnp.exp(jnp.where(is_target, -jnp.inf, safe_logits - dense_lse[:, None])), axis=-1)
    dense_zero_loss = dense_ce == 0
    dense_zero_gradient = dense_target_gradient == 0
    dense_collapse = dense_zero_loss & dense_zero_gradient & (non_target_mass > 0)

    def count(condition: jax.Array) -> jax.Array:
        return jnp.sum(condition, dtype=jnp.int32)

    def total(value: jax.Array) -> jax.Array:
        return jnp.sum(jnp.where(valid, value, 0.0))

    return {
        "input_count": jnp.asarray(logits.shape[0], dtype=jnp.int32),
        "masked_count": count(weights == 0),
        "invalid_weight_count": count(~finite_weight | (weights < 0)),
        "nonfinite_logits_count": count(positive_weight & ~finite_logits),
        "invalid_target_count": count(positive_weight & ~valid_target),
        "valid_count": count(valid),
        "weight_sum": total(weights),
        "margin_above_fp32_cliff_count": count(valid & (margin > FP32_MARGIN_CLIFF)),
        "dense_fp32_zero_loss_count": count(valid & dense_zero_loss),
        "dense_fp32_zero_target_gradient_count": count(valid & dense_zero_gradient),
        "dense_fp32_collapse_count": count(valid & dense_collapse),
        "dense_fp32_collapse_weight_sum": total(jnp.where(dense_collapse, weights, 0.0)),
        "target_argmax_count": count(valid & (jnp.argmax(safe_logits, axis=-1) == safe_targets)),
        "stable_ce_sum": total(stable_ce),
        "stable_target_gradient_sum": total(stable_target_gradient),
        "dense_fp32_ce_sum": total(dense_ce),
        "dense_fp32_target_gradient_sum": total(dense_target_gradient),
        "dense_fp32_gradient_imbalance_sum": total(dense_target_gradient + non_target_mass),
        "dense_fp32_ce_absolute_error_sum": total(jnp.abs(dense_ce - stable_ce)),
        "dense_fp32_target_gradient_absolute_error_sum": total(jnp.abs(dense_target_gradient - stable_target_gradient)),
        "max_logit_sum": total(jnp.max(safe_logits, axis=-1)),
        "min_logit_sum": total(jnp.min(safe_logits, axis=-1)),
        "target_logit_sum": total(target_logit),
        "dense_fp32_lse_sum": total(dense_lse),
        "margin_sum": total(margin),
        "margin_histogram": jnp.stack(
            [count(valid & (margin >= lo) & (margin < hi)) for lo, hi in pairwise(MARGIN_HISTOGRAM_EDGES)]
        ),
    }
