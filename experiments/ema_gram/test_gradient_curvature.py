# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from levanter.layers.attention import AttentionBackend
from levanter.models.lm_model import LmExample
from levanter.models.qwen import QwenConfig
from levanter.optim.ema_gram import EmaGramConfig
from levanter.optim.muon import MuonConfig

from experiments.ema_gram.gradient_curvature import (
    PathMeasurement,
    measurement_functions,
    production_functions,
    production_updates,
)
from experiments.ema_gram.gradient_curvature_math import predicted_loss, tree_norm, weighted_gradient


def test_weighted_vectors_and_independent_quartic_path_integral():
    weights = np.arange(1, 9, dtype=float)
    batches = list(np.linspace(-2, 2, 8))
    model = jnp.array([0.7, -0.4])
    direction = jnp.array([0.6, 0.8])

    def loss(x, b):
        return jnp.sum(0.25 * x**4 - x**2) + b * jnp.sum(x)

    grad = jax.grad(loss)
    value_grad = jax.jit(jax.value_and_grad(loss))
    product = jax.jit(lambda x, d, b: jax.jvp(lambda y: grad(y, b), (x,), (d,))[1])
    engine = PathMeasurement(model, batches, weights, [8], value_grad, product, loss, lambda _: None)
    length = 0.3
    points = [engine.point(direction, length * u)["8"] for u in (0, 0.5, 1)]
    mean_b = np.average(batches, weights=weights)
    expected_grad = np.array(model) ** 3 - 2 * np.array(model) + mean_b
    np.testing.assert_allclose(points[0]["G"], np.linalg.norm(expected_grad), rtol=1e-6)
    assert points[0]["kappa"] < 0
    predicted = predicted_loss(length, points[0]["alignment"], [p["kappa"] for p in points])
    expected = float(loss(model - length * direction, mean_b) - loss(model, mean_b))
    np.testing.assert_allclose(predicted, expected, atol=1e-7)
    check = engine.derivative_check(direction, 0)
    assert check["passed"] and len(check["checks"]) >= 2
    _, cancelled = weighted_gradient(
        jnp.array([0.0]), [-1.0, 1.0], [1.0, 1.0], jax.jit(jax.value_and_grad(lambda x, b: b * x[0]))
    )
    assert tree_norm(cancelled) == 0


def test_real_model_full_hvp_and_production_matrix_control():
    config = QwenConfig(
        hidden_dim=8,
        intermediate_dim=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=8,
        attn_backend=AttentionBackend.VANILLA,
    )
    model = config.build(hax.Axis("vocab", 16), key=jax.random.PRNGKey(2))
    batch = LmExample.causal(hax.named(jnp.array([1, 4, 2, 3, 6, 2, 1, 0]), config.max_Pos))
    optimizer = EmaGramConfig(learning_rate=0.001, warmup=0.0, alpha=0.125, ridge=0.01, weight_decay=0.1)
    state = SimpleNamespace(opt_state=optimizer.build(100).init(eqx.filter(model, eqx.is_inexact_array)))
    cfg = SimpleNamespace(
        optimizer=optimizer,
        z_loss_weight=1e-7,
        trainer=SimpleNamespace(
            num_train_steps=100,
            compute_axis_mapping={},
            mp=jmp.get_policy("params=float32,compute=float32,output=float32"),
        ),
    )
    before = [np.array(x).copy() for x in jax.tree.leaves((model, state.opt_state)) if eqx.is_array(x)]
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(-1, 1), ("data", "model"))
    with jax.sharding.set_mesh(mesh):
        value_grad, hvp, _ = measurement_functions({})
        prod_grad, functions = production_functions(cfg)
        native_loss, gradient = prod_grad(model, batch)
        reference_loss, _ = value_grad(model, batch)
        np.testing.assert_allclose(native_loss, reference_loss, rtol=2e-6)
        full, _ = production_updates(state, model, gradient, functions["full", 0.125], False)
        matrix, _ = production_updates(state, model, gradient, functions["matrix_only", 0.125], True)
        mask = MuonConfig().create_mask(model, use_kimi_scaling=True)
        for label, update in zip(jax.tree.leaves(mask), jax.tree.leaves(matrix), strict=True):
            if label != "muon":
                np.testing.assert_array_equal(update, np.zeros_like(update))
        direction = jax.tree.map(lambda x: x / tree_norm(full), full)
        actual = hvp(model, direction, batch)
        eps = 0.003
        plus = jax.tree.map(lambda x, d: x + eps * d, model, direction)
        minus = jax.tree.map(lambda x, d: x - eps * d, model, direction)
        _, gp = value_grad(plus, batch)
        _, gm = value_grad(minus, batch)
        fd = jax.tree.map(lambda a, b: (a - b) / (2 * eps), gp, gm)
        err = tree_norm(jax.tree.map(lambda a, b: a - b, actual, fd)) / tree_norm(actual)
        assert err < 0.01
    after = [np.array(x) for x in jax.tree.leaves((model, state.opt_state)) if eqx.is_array(x)]
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(a, b)
