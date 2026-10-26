# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from experiments.grug.fast_track.dense_optimizers import DenseOptimizer, DenseOptimizerConfig


def _parameters():
    return {
        "blocks": {
            "attn": {"w_q": jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 20 + 0.1},
            "mlp": {"w_gate": jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 30 + 0.1},
            "rms_attn": {"weight": jnp.ones((2, 3))},
        },
        "embed_gated_norm": {"w_down": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 10 + 0.1},
        "output_proj": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 15 + 0.1,
        "token_embed": jnp.ones((5, 3)),
    }


def test_dense_adamw_matches_optax_with_norm_decay_excluded():
    params = _parameters()
    config = DenseOptimizerConfig(
        optimizer=DenseOptimizer.ADAMW,
        learning_rate=0.01,
        adam_lr=0.01,
        weight_decay=0.1,
        warmup=0,
        lr_schedule="constant",
    )
    candidate = config.build(10)
    decay_mask = jax.tree.map(lambda _: True, params)
    decay_mask["blocks"]["rms_attn"]["weight"] = False
    reference = optax.adamw(0.01, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1, mask=decay_mask)
    candidate_state = candidate.init(params)
    reference_state = reference.init(params)
    for step in range(3):
        gradients = jax.tree.map(lambda p, step=step: jnp.sin(p * (step + 1)), params)
        actual, candidate_state = candidate.update(gradients, candidate_state, params)
        expected, reference_state = reference.update(gradients, reference_state, params)
        for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=2e-6, atol=1e-7)
        params = optax.apply_updates(params, actual)


def test_dense_adamh_first_step_matches_normalized_adam_projection():
    params = {"w_q": _parameters()["blocks"]["attn"]["w_q"]}
    gradients = jax.tree.map(lambda p: jnp.cos(p * 3), params)
    transform = DenseOptimizerConfig(
        optimizer=DenseOptimizer.ADAMH,
        learning_rate=0.03,
        weight_decay=0,
        warmup=0,
        lr_schedule="constant",
    ).build(10)
    updates, _ = transform.update(gradients, transform.init(params), params)
    matrix = np.asarray(params["w_q"])
    direction = np.asarray(gradients["w_q"])
    direction = direction / (np.abs(direction) + 1e-8)
    norms = np.linalg.norm(matrix, axis=(-2, -1), keepdims=True)
    expected = matrix - 0.03 * direction * norms / np.linalg.norm(direction, axis=(-2, -1), keepdims=True)
    expected *= norms / np.linalg.norm(expected, axis=(-2, -1), keepdims=True)
    np.testing.assert_allclose(matrix + updates["w_q"], expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("optimizer", [DenseOptimizer.ADAMH, DenseOptimizer.MUONH])
def test_dense_hyperball_updates_preserve_each_matrix_norm_and_move_fallback(optimizer):
    params = _parameters()
    transform = DenseOptimizerConfig(
        optimizer=optimizer,
        learning_rate=0.03,
        adam_lr=0.01,
        weight_decay=0,
        warmup=0,
        lr_schedule="constant",
    ).build(10)
    gradients = jax.tree.map(lambda p: jnp.cos(p * 3), params)
    updates, _ = transform.update(gradients, transform.init(params), params)
    updated = optax.apply_updates(params, updates)
    for path in (
        ("blocks", "attn", "w_q"),
        ("blocks", "mlp", "w_gate"),
        ("embed_gated_norm", "w_down"),
        ("output_proj",),
    ):
        before, after = params, updated
        for key in path:
            before, after = before[key], after[key]
        np.testing.assert_allclose(
            jnp.linalg.norm(after, axis=(-2, -1)), jnp.linalg.norm(before, axis=(-2, -1)), rtol=2e-6, atol=1e-7
        )
        assert float(jnp.linalg.norm(after - before)) > 0
    np.testing.assert_allclose(updated["token_embed"], 1.01, atol=1e-6)
    assert not np.allclose(updated["blocks"]["rms_attn"]["weight"], params["blocks"]["rms_attn"]["weight"])


@pytest.mark.parametrize("optimizer", [DenseOptimizer.MUON, DenseOptimizer.MUONH])
def test_dense_muon_stacked_matrices_match_independent_matrix_updates(optimizer):
    config = DenseOptimizerConfig(
        optimizer=optimizer,
        learning_rate=0.03,
        adam_lr=0.01,
        weight_decay=0,
        warmup=0,
        lr_schedule="constant",
    )
    stacked = {"w_q": _parameters()["blocks"]["attn"]["w_q"]}
    individual = {f"w_q_{i}": matrix for i, matrix in enumerate(stacked["w_q"])}
    transform = config.build(10)
    stacked_gradients = jax.tree.map(lambda p: jnp.sin(p * 2), stacked)
    individual_gradients = jax.tree.map(lambda p: jnp.sin(p * 2), individual)
    actual, _ = transform.update(stacked_gradients, transform.init(stacked), stacked)
    expected, _ = transform.update(individual_gradients, transform.init(individual), individual)
    np.testing.assert_allclose(actual["w_q"], jnp.stack(list(expected.values())), rtol=1e-5, atol=1e-7)
