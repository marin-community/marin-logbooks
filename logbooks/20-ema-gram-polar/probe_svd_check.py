# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tiny CPU oracle for the production attention/MLP decomposition diagnostics."""

from dataclasses import asdict

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from levanter.models.qwen import Qwen3Config
from levanter.optim.ema_gram import SVD, EmaGramConfig, EmaGramState, GramState, exact_polar, gram_direction
from levanter.optim.util import flatten_linear_layers
from levanter.trainer_state import TrainerState

from experiments.ema_gram.probe import decomposition_diagnostics, matrix_state


def numpy_polar(matrix):
    u, singular, vh = np.linalg.svd(matrix.astype(np.float64), full_matrices=False)
    threshold = np.finfo(np.float32).eps * max(matrix.shape) * singular[:1]
    return (u * (singular > threshold)) @ vh


def numpy_power(gram, alpha, ridge):
    gram = gram.astype(np.float64)
    normalized = (gram + gram.T) * 0.5 / (np.trace(gram) / len(gram))
    values, vectors = np.linalg.eigh(normalized)
    return (vectors * (np.maximum(values, 0) + ridge) ** (-alpha)) @ vectors.T


def execute():
    assert jax.default_backend() == "cpu"
    cfg = Qwen3Config(
        hidden_dim=8, intermediate_dim=16, num_layers=2, num_heads=2, num_kv_heads=2, head_dim=4, max_seq_len=8
    )
    model = cfg.build(hax.Axis("vocab", 16), key=jax.random.PRNGKey(1))
    state = TrainerState.init(EmaGramConfig(alpha=0.25).build(20), model, key=jax.random.PRNGKey(2))
    _, _, optimizer = matrix_state(state.model, state.opt_state)
    rng = np.random.default_rng(317)
    matrices = []
    for matrix in optimizer.matrices:
        momentum = rng.normal(size=matrix.momentum.shape).astype(np.float32)
        grams = []
        for template in (matrix.left, matrix.right):
            a = rng.normal(size=template.shape).astype(np.float32)
            grams.append(jnp.asarray(a @ a.swapaxes(-1, -2) + np.eye(a.shape[-1], dtype=np.float32)))
        matrices.append(GramState(jnp.asarray(momentum), *grams))
    replacement = EmaGramState(tuple(matrices))
    opt_state = jax.tree.map(
        lambda x: replacement if isinstance(x, EmaGramState) else x,
        state.opt_state,
        is_leaf=lambda x: isinstance(x, EmaGramState),
    )
    before = [np.asarray(x).copy() for x in jax.tree.leaves(opt_state)]
    rows = decomposition_diagnostics(state.model, opt_state, polar_steps=5)
    assert len(rows) == 18 and {r.category for r in rows} == {"attention", "mlp"}
    masked, _, optimizer = matrix_state(state.model, opt_state)
    paths = jax.tree_util.tree_flatten_with_path(flatten_linear_layers(masked))[0]
    by_path = {
        jax.tree_util.keystr(path): GramState(*(x.reshape((-1, *x.shape[-2:]))[0] for x in matrix))
        for (path, _), matrix in zip(paths, optimizer.matrices, strict=True)
    }
    errors = []
    for row in rows:
        matrix = by_path[row.path]
        momentum = np.asarray(matrix.momentum)
        if row.alpha:
            left, right = (numpy_power(np.asarray(x), row.alpha, row.ridge) for x in (matrix.left, matrix.right))
            expected = left @ numpy_polar(left @ momentum @ right) @ right
        else:
            expected = numpy_polar(momentum)
        exact = np.asarray(gram_direction(matrix, alpha=row.alpha, ridge=row.ridge, polar_steps=5, polar_method=SVD))
        approx = np.asarray(gram_direction(matrix, alpha=row.alpha, ridge=row.ridge, polar_steps=5))
        np.testing.assert_allclose(exact, expected, rtol=2e-4, atol=2e-5)
        cosine = float(np.vdot(approx, expected) / (np.linalg.norm(approx) * np.linalg.norm(expected)))
        relative = float(np.linalg.norm(approx - expected) / np.linalg.norm(expected))
        np.testing.assert_allclose([row.cosine, row.relative_error], [cosine, relative], rtol=2e-4, atol=2e-5)
        assert all(np.isfinite(v) and v >= 0 for v in row.seconds.values())
        errors.append(float(np.max(np.abs(exact - expected))))
    for a, b in zip(before, jax.tree.leaves(opt_state), strict=True):
        np.testing.assert_array_equal(a, np.asarray(b))
    edge_cases = [np.zeros((4, 3), np.float32), np.ones((4, 3), np.float32), np.ones((3, 4), np.float32)]
    for matrix in edge_cases:
        np.testing.assert_allclose(
            np.asarray(exact_polar(jnp.asarray(matrix))), numpy_polar(matrix), rtol=1e-5, atol=1e-5
        )
    return dict(
        success=True,
        rows=[asdict(r) for r in rows],
        max_svd_absolute_error=max(errors),
        zero_and_rank_deficient_cases=len(edge_cases),
        state_immutable=True,
        scope="Tiny CPU numerical oracle only; timings are not an accelerator performance result; no training updates.",
        seeds=dict(model=1, state=2, numpy=317),
    )
