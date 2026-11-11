# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from levanter.layers.attention import AttentionBackend
from levanter.models.lm_model import LmExample
from levanter.models.qwen import QwenConfig
from levanter.optim.util import flatten_linear_layers

from experiments.ema_gram.curvature_resolution import projected_resolution
from experiments.ema_gram.frozen_controls import effective_ema_samples, project_grams, rank_one_factors
from experiments.ema_gram.hessian import block_functions, projected_difference_column, select_basis


def test_block_hvp_matches_dense_hessian_and_native_loss():
    cfg = QwenConfig(
        hidden_dim=8,
        intermediate_dim=16,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=8,
        attn_backend=AttentionBackend.VANILLA,
    )
    model = cfg.build(hax.Axis("vocab", 16), key=jax.random.PRNGKey(8))
    tokens = hax.named(jnp.array([1, 4, 3, 2, 5, 3, 1, 0]), cfg.max_Pos)
    batch = LmExample.causal(tokens)
    paths = jax.tree_util.tree_flatten_with_path(flatten_linear_layers(model))[0]
    for path, bank in paths:
        name = jax.tree_util.keystr(path)
        if "self_attn.q_proj.weight" not in name and "mlp.gate_proj.weight" not in name:
            continue
        matrix = bank.reshape((-1, *bank.shape[-2:]))[0]
        loss, grad, hvp = block_functions(model, name, 1e-7)
        expected = model.compute_next_token_loss(batch, logsumexp_weight=1e-7).scalar()
        np.testing.assert_allclose(loss(matrix, model, batch), expected, rtol=1e-6)
        direction = jax.random.normal(jax.random.PRNGKey(7), matrix.shape)
        direction /= jnp.linalg.norm(direction)
        dense = jax.hessian(lambda w, loss=loss: loss(w, model, batch))(matrix)
        expected_hvp = jnp.einsum("ijkl,kl->ij", dense, direction)
        actual = hvp(matrix, direction, model, batch)
        np.testing.assert_allclose(actual, expected_hvp, rtol=2e-4, atol=2e-6)
        delta = 3e-3
        fd = (grad(matrix + delta * direction, model, batch) - grad(matrix - delta * direction, model, batch)) / (
            2 * delta
        )
        np.testing.assert_allclose(actual, fd, rtol=0.02, atol=2e-5)


def test_selected_basis_matches_kronecker_operator():
    rng = np.random.default_rng(17)
    u, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    v, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    left = (u * np.geomspace(0.1, 2, 6)) @ u.T
    right = (v * np.geomspace(0.2, 3, 4)) @ v.T
    directions, raw, ridged, _ = select_basis(left, right, 12)
    np.testing.assert_allclose(np.einsum("iab,jab->ij", directions, directions), np.eye(12), atol=2e-7)
    for i, e in enumerate(directions):
        np.testing.assert_allclose(left @ e @ right, raw[i, i] * e, rtol=1e-5, atol=2e-7)
        a = left / (np.trace(left) / 6) + 0.01 * np.eye(6)
        b = right / (np.trace(right) / 4) + 0.01 * np.eye(4)
        np.testing.assert_allclose(a @ e @ b, ridged[i, i] * e, rtol=1e-5, atol=2e-7)
    wider, _, _, _ = select_basis(left, right, 20)
    np.testing.assert_array_equal(wider[:12], directions)


def test_fresh_gram_projection_preserves_off_diagonal_entries():
    rng = np.random.default_rng(7)
    basis, _, _, _ = select_basis(np.diag([1.0, 2, 4]), np.diag([1.0, 3]), 5)
    u, v = rank_one_factors(basis)
    x, y = rng.normal(size=(3, 3)), rng.normal(size=(2, 2))
    left, right = x @ x.T + np.eye(3), y @ y.T + np.eye(2)
    raw, ridged = project_grams(left, right, u, v)
    explicit = np.array([[np.sum(e * (left @ f @ right)) for f in basis] for e in basis])
    np.testing.assert_allclose(raw, explicit, rtol=1e-6)
    assert np.linalg.norm(raw - np.diag(np.diag(raw))) > 0.1
    a, b = left / (np.trace(left) / 3) + 0.01 * np.eye(3), right / (np.trace(right) / 2) + 0.01 * np.eye(2)
    explicit = np.array([[np.sum(e * (a @ f @ b)) for f in basis] for e in basis])
    np.testing.assert_allclose(ridged, explicit, rtol=1e-6)
    assert 196 < effective_ema_samples(0.99, 512) < 199


def test_resolution_screen_respects_scale_and_rejects_uncertain_curvature():
    mean = np.diag([1e-8, 1.0, 1e4])
    batches = np.stack([mean, mean])
    references = np.stack([batches * 1.001, batches * 0.999])
    result = projected_resolution(batches, references)
    assert result["status"] == "empirically_resolved"
    assert result["margin_ratio"] > 400
    uncertain = references.copy()
    uncertain[0, :, 0, 0] = 0.4e-8
    assert projected_resolution(batches, uncertain)["status"] == "unresolved"
    indefinite = batches.copy()
    indefinite[:, 0, 0] = -1e-8
    assert projected_resolution(indefinite, np.stack([indefinite, indefinite]))["status"] == "indefinite"


def test_projected_finite_differences_preserve_mixing_and_reject_wrong_hvp():
    operator = jnp.array([[2.0, 0.4], [0.4, 3.0]])
    matrix = jnp.array([[0.2, 0.7]])
    basis = np.eye(2, dtype=np.float32).reshape(2, 1, 2)

    def loss(w, model, batch):
        return (w.ravel() @ operator @ w.ravel()) / 2

    grad = jax.grad(loss)
    actual = np.asarray(operator[:, 0]).reshape(1, 2)
    check, references = projected_difference_column(matrix, basis[0], None, None, grad, actual, basis, None)
    assert check["passed"]
    np.testing.assert_allclose(references, np.tile(np.asarray(operator[:, 0]), (2, 1)), rtol=2e-6)
    rejected, reference = projected_difference_column(matrix, basis[0], None, None, grad, actual * 2, basis, None)
    assert not rejected["passed"] and reference is None
