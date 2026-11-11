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

from experiments.ema_gram.hessian import block_functions, select_basis


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
