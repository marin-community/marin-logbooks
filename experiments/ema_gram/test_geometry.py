# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from itertools import pairwise
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

from experiments.ema_gram.geometry_math import (
    envelope_components,
    normalized_hessian_norm,
    orthonormal_matrices,
    quadratic_step,
    refinement_lengths,
    rotated_gradient,
)
from experiments.ema_gram.gradient_curvature import production_functions
from experiments.ema_gram.gradient_curvature_math import tree_add, tree_norm, tree_scale
from experiments.ema_gram.power_prediction import ALPHA_FD_STEPS, POWERS, fp32_direction_function, run_power_prediction
from experiments.ema_gram.probe import matrix_state


def test_bounded_quadratic_handles_negative_curvature_and_zero_action():
    for a, k, expected in [(2, 4, 0.5), (-2, 4, 0), (2, -4, 2), (-3, -4, 2), (-5, -4, 0), (0, 0, 0)]:
        step, value = quadratic_step(a, k, 2)
        assert step == expected
        grid = np.linspace(0, 2, 10001)
        assert value >= np.max(a * grid - 0.5 * k * grid**2) - 1e-12
    assert refinement_lengths([0, 1, 3], [4, 2, 3]) == ([0.5, 2.0], False)
    assert refinement_lengths([0, 1, 3], [4, 3, 2]) == ([], True)


def test_matrix_envelope_matches_dense_operator_and_detects_off_diagonal_violation():
    rng = np.random.default_rng(9)
    g = rng.normal(size=(3, 2))
    basis, _ = orthonormal_matrices([(str(i), rng.normal(size=g.shape)) for i in range(4)], 4)
    c = envelope_components(g, basis)
    dense = 0.1 * np.eye(6) + 0.3 * np.kron(g @ g.T, np.eye(2)) + 0.7 * np.kron(np.eye(3), g.T @ g)
    v = basis.reshape(4, -1).T
    np.testing.assert_allclose(np.einsum("i,ijk->jk", [0.1, 0.3, 0.7], c), v.T @ dense @ v, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.svd(rotated_gradient(g, 19), compute_uv=False), np.linalg.svd(g, compute_uv=False), atol=1e-12
    )
    # Every diagonal passes, while the coupled eigenmode violates the bound.
    assert normalized_hessian_norm(np.array([[0.5, 1.0], [1.0, 0.5]]), np.eye(2)) == 1.5


def test_fp32_alpha_tangent_matches_complete_production_update_and_finite_difference():
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
    cfg = SimpleNamespace(
        optimizer=optimizer,
        z_loss_weight=1e-7,
        trainer=SimpleNamespace(
            num_train_steps=100,
            compute_axis_mapping={},
            mp=jmp.get_policy("params=float32,compute=float32,output=float32"),
        ),
    )
    state = optimizer.build(100).init(eqx.filter(model, eqx.is_inexact_array))
    before = [np.array(x).copy() for x in jax.tree.leaves(state) if eqx.is_array(x)]
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(-1, 1), ("data", "model"))
    with jax.sharding.set_mesh(mesh):
        prod_grad, functions = production_functions(cfg, alphas=POWERS, modes=("full",))
        _, gradient = prod_grad(model, batch)
        middle, after = functions["full", 0.1875](gradient, state, eqx.filter(model, eqx.is_inexact_array))
        smooth = fp32_direction_function(cfg)
        for alpha in POWERS:
            native, other = functions["full", alpha](gradient, state, eqx.filter(model, eqx.is_inexact_array))
            np.testing.assert_allclose(
                tree_norm(tree_add(matrix_state(model, after).optimizer, matrix_state(model, other).optimizer, -1)),
                0,
                atol=0,
            )
            direct = tree_scale(native, -1 / tree_norm(native))
            actual = smooth(model, after, middle, jnp.array(alpha))
            assert tree_norm(tree_add(actual, direct, -1)) < 2e-5
        alpha = jnp.array(0.1875)
        _, tangent = jax.jvp(lambda a: smooth(model, after, middle, a), (alpha,), (jnp.ones_like(alpha),))
        errors = []
        for eps in ALPHA_FD_STEPS:
            fd = tree_scale(
                tree_add(smooth(model, after, middle, alpha + eps), smooth(model, after, middle, alpha - eps), -1),
                1 / (2 * eps),
            )
            errors.append(tree_norm(tree_add(fd, tangent, -1)) / tree_norm(tangent))
        assert any(a < 0.02 and b < 0.02 for a, b in pairwise(errors)), errors
        saved = {}
        events = []

        def save(name, value):
            saved[name] = json.loads(json.dumps(value))

        result = run_power_prediction(
            state=SimpleNamespace(opt_state=state, step=0),
            model=model,
            native_model=model,
            cfg=cfg,
            panels={"calibration": [batch] * 2, "evaluation": [batch] * 2, "update": [batch] * 2},
            weights={"calibration": [1, 1], "evaluation": [1, 1]},
            args=SimpleNamespace(state="tiny"),
            protocol={"length_grid": [0, 0.1, 0.3, 0.6, 1, 2, 4]},
            save=save,
            progress=events.append,
            manifest_hash="unit-test",
            save_arrays=lambda *a, **k: None,
        )
        assert len(result["rows"]) == 6
        assert all("evaluation" not in row for row in saved["locked-predictions.json"]["rows"])
        assert [e["event"] for e in events].index("predictions_locked") < [e["event"] for e in events].index(
            "heldout_direction"
        )
        for row in result["rows"]:
            chosen = next(v for v in row["calibration"] if v["length"] == row["direct_step"])
            np.testing.assert_allclose(chosen["losses"], row["evaluation"]["direct"]["losses"], atol=1e-7)
    for old, new in zip(before, [np.array(x) for x in jax.tree.leaves(state) if eqx.is_array(x)], strict=True):
        np.testing.assert_array_equal(old, new)
