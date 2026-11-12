# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Vector aggregation and independent path-curvature integration for frozen probes."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


@eqx.filter_jit
def tree_dot(left, right):
    return sum(jnp.vdot(a, b).real for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True))


@eqx.filter_jit
def tree_scale(tree, scale):
    return jax.tree.map(lambda x: x * scale, tree)


@eqx.filter_jit
def tree_add(left, right, scale):
    return jax.tree.map(lambda a, b: a + scale * b, left, right)


def tree_norm(tree):
    return float(jnp.sqrt(tree_dot(tree, tree)))


def weighted_gradient(model, batches, weights, value_grad):
    """Average vectors with valid-token weights before computing any vector norm."""
    total = None
    losses = []
    normalized = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    for batch, weight in zip(batches, normalized, strict=True):
        loss, gradient = value_grad(model, batch)
        losses.append(float(loss))
        total = tree_scale(gradient, weight) if total is None else tree_add(total, gradient, weight)
    return losses, total


def curvature_integral(values):
    """Composite Simpson integral of (1-u) kappa(u), independent of loss values."""
    values = np.asarray(values, dtype=np.float64)
    intervals = len(values) - 1
    if intervals < 2 or intervals % 2:
        raise ValueError("Simpson integration requires a positive even number of intervals")
    weights = np.ones(len(values))
    weights[1:-1:2] = 4
    weights[2:-1:2] = 2
    return float(np.sum(weights * (1 - np.linspace(0, 1, len(values))) * values) / (3 * intervals))


def predicted_loss(length, alignment, curvature):
    return -length * alignment + length**2 * curvature_integral(curvature)


def paired_interval(differences, valid_weights, seed=20261112, draws=10000):
    """Crossed resampling of update draws and shared reference groups, not path points."""
    differences = np.asarray(differences, dtype=np.float64)
    weights = np.asarray(valid_weights, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(differences.shape[0], size=(draws, differences.shape[0]))
    group_indices = rng.integers(differences.shape[1], size=(draws, differences.shape[1]))
    sampled = differences[draw_indices[:, :, None], group_indices[:, None, :]]
    sampled_weights = weights[group_indices]
    means = np.mean(np.sum(sampled * sampled_weights[:, None, :], axis=2) / sampled_weights.sum(axis=1)[:, None], axis=1)
    return dict(
        mean=float(np.mean(differences @ weights / weights.sum())),
        lower=float(np.quantile(means, 0.025)),
        upper=float(np.quantile(means, 0.975)),
    )
