# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from levanter.models.qwen import QwenConfig

from experiments.ema_gram.gradient_curvature_math import tree_add, tree_norm
from experiments.ema_gram.muon_geometry_kernels import (
    correction_telemetry,
    f5_finite_difference,
    matrix_candidates,
)
from experiments.ema_gram.muon_geometry_updates import (
    add_f5_feedback,
    momentum_state,
    native_muon_functions,
    tangent_functions,
)
from experiments.speedrun.prism_berkeley_qwen3_scaling.muon_error_feedback_optimizer import (
    ErrorAwareMuonConfig,
    NuclearHessianSettings,
    clipped_nuclear_hessian,
    quintic_newton_schulz,
)


@pytest.mark.parametrize("shape", [(8, 5), (5, 8)])
def test_muon_controls_match_pinned_map_and_tangent_finite_differences(shape):
    rng = np.random.default_rng(72)
    momentum = jnp.asarray(rng.normal(size=shape), jnp.float32)
    gradient = jnp.asarray(rng.normal(size=shape), jnp.float32)
    settings = NuclearHessianSettings()
    candidates, tangent, metrics = jax.jit(lambda m, g: matrix_candidates(m, g, settings))(momentum, gradient)
    nesterov = 0.98 * momentum + 0.02 * gradient
    np.testing.assert_allclose(candidates["blend_0.02"], quintic_newton_schulz(nesterov), rtol=1e-5, atol=1e-6)
    correction, _ = correction_telemetry(momentum, gradient - momentum, settings)
    np.testing.assert_allclose(
        correction, clipped_nuclear_hessian(momentum, gradient - momentum, settings=settings), rtol=1e-5, atol=1e-6
    )
    errors = []
    for epsilon in (0.003, 0.001):
        fd, _ = f5_finite_difference(momentum, gradient, epsilon)
        errors.append(float(jnp.linalg.norm(fd - tangent) / jnp.linalg.norm(tangent)))
    assert max(errors) < 0.02
    assert float(metrics["reconstructed_hesscorr_error"]) < 1e-5
    assert bool(metrics["solve_finite"])
    assert 0 <= float(metrics["clipping_factor"]) <= 1


def test_radial_correction_zero_is_distinguished_from_nonfinite_fallback():
    matrix = jnp.eye(4)
    correction, metrics = correction_telemetry(matrix, 3 * matrix, NuclearHessianSettings())
    np.testing.assert_array_equal(correction, 0)
    assert bool(metrics["radial_zero"]) and not bool(metrics["nonfinite_zero"])


def test_complete_muon_updates_share_moments_and_keep_adam_in_tangent_control():
    model_config = QwenConfig(
        hidden_dim=8, intermediate_dim=16, num_layers=2, num_heads=2, num_kv_heads=2, max_seq_len=8
    )
    model = model_config.build(hax.Axis("vocab", 32), key=jax.random.PRNGKey(16))
    optimizer = ErrorAwareMuonConfig(
        learning_rate=0.01, adam_lr=0.003, momentum=0.98, policy="muon", correction_warmup_steps=0
    )
    cfg = SimpleNamespace(
        optimizer=optimizer,
        z_loss_weight=None,
        trainer=SimpleNamespace(
            num_train_steps=100,
            compute_axis_mapping={},
            mp=jmp.get_policy("params=float32,compute=float32,output=float32"),
        ),
    )
    state = optimizer.build(100).init(eqx.filter(model, eqx.is_inexact_array))
    _, updates, _ = native_muon_functions(cfg, 0.012)
    tangent_fn, difference = tangent_functions(cfg)
    rng = np.random.default_rng(59)

    def gradient():
        return jax.tree.map(lambda x: jnp.asarray(rng.normal(size=x.shape), x.dtype) * 0.01, model)

    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(-1, 1), ("data", "model"))
    with jax.sharding.set_mesh(mesh):
        _, state = updates["muon"](gradient(), state, model)
        g = gradient()
        plain, post = updates["muon"](g, state, model)
        derivative, _ = tangent_fn(post, g)
        tangent_update = add_f5_feedback(plain, derivative, 0.012)
        for function in updates.values():
            candidate, advanced = function(g, state, model)
            assert np.isfinite(float(optax.global_norm(candidate)))
            assert tree_norm(tree_add(momentum_state(post), momentum_state(advanced), -1)) == 0
        for epsilon in (0.003, 0.001):
            fd = difference(post, g, epsilon)
            assert tree_norm(tree_add(fd, derivative, -1)) / tree_norm(derivative) < 0.02
    np.testing.assert_allclose(float(post.hyperparams["learning_rate"]), 0.012)
    np.testing.assert_allclose(float(post.hyperparams["adam_lr"]), 0.0036)
    masked = momentum_state(post).momentum_buffer
    # The feedback adds only matrix leaves; embeddings and all other Adam leaves stay exact.
    checked = []

    def assert_same(t, a, b):
        if isinstance(t, optax.MaskedNode):
            for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
                np.testing.assert_array_equal(left, right)
                checked.append(True)
        return None

    jax.tree.map(assert_same, masked, plain, tangent_update, is_leaf=lambda x: isinstance(x, optax.MaskedNode))
    assert len(checked) >= 2
