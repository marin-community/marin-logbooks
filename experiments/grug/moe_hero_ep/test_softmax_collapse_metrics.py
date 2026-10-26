# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.moe_hero_ep.softmax_collapse_metrics import softmax_collapse_metrics


def test_softmax_collapse_metrics_preserve_tiny_ce_and_target_gradient():
    metrics = jax.jit(softmax_collapse_metrics)(
        jnp.array([[18.0, 0.0, 0.0], [0.0, 18.0, 0.0]]), jnp.array([0, 1]), jnp.ones(2)
    )
    expected_ce = math.log1p(2 * math.exp(-18))
    expected_gradient = -2 * math.exp(-18) / (1 + 2 * math.exp(-18))
    assert metrics["dense_fp32_collapse_count"] == 2
    assert metrics["margin_above_fp32_cliff_count"] == 2
    np.testing.assert_allclose(metrics["stable_ce_sum"], 2 * expected_ce, rtol=1e-5, atol=0)
    np.testing.assert_allclose(metrics["stable_target_gradient_sum"], 2 * expected_gradient, rtol=1e-5, atol=0)
    assert metrics["dense_fp32_target_gradient_sum"] == 0
    assert metrics["dense_fp32_gradient_imbalance_sum"] > 0


def test_softmax_collapse_metrics_include_entire_vocabulary_tail():
    # A gap of 18 to the runner-up is insufficient when 99,999 competitors contribute.
    logits = jnp.zeros((1, 100_000)).at[0, 0].set(18)
    metrics = jax.jit(softmax_collapse_metrics)(logits, jnp.array([0]), jnp.ones(1))
    assert metrics["margin_above_fp32_cliff_count"] == 0
    assert metrics["dense_fp32_collapse_count"] == 0
    np.testing.assert_allclose(metrics["margin_sum"], 18 - math.log(99_999), rtol=1e-5, atol=0)
    assert metrics["stable_ce_sum"] > 0.001


def test_softmax_collapse_metrics_mask_nonfinite_rows_and_add_across_blocks():
    logits = jnp.array([[18.0, 0.0, 0.0], [jnp.nan, 0.0, 0.0], [jnp.inf, 0.0, 0.0], [0.0, 0.0, 0.0], [18.0, 0.0, 0.0]])
    targets = jnp.array([0, 0, 0, 5, 0])
    weights = jnp.array([2.0, 0.0, 1.0, 1.0, jnp.nan])
    metrics = jax.jit(softmax_collapse_metrics)(logits, targets, weights)
    assert metrics["valid_count"] == 1
    assert metrics["masked_count"] == 1
    assert metrics["nonfinite_logits_count"] == 1
    assert metrics["invalid_target_count"] == 1
    assert metrics["invalid_weight_count"] == 1
    assert metrics["dense_fp32_collapse_count"] == 1
    assert metrics["dense_fp32_collapse_weight_sum"] == 2
    assert metrics["weight_sum"] == 2
    assert metrics["margin_histogram"].sum() == 1
    first = softmax_collapse_metrics(logits[:2], targets[:2], weights[:2])
    second = softmax_collapse_metrics(logits[2:], targets[2:], weights[2:])
    for name, value in metrics.items():
        assert np.isfinite(np.asarray(value)).all(), name
        np.testing.assert_allclose(value, first[name] + second[name], rtol=1e-5, atol=0)
