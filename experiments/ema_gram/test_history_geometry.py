# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from levanter.models.qwen import QwenConfig
from levanter.optim.ema_gram import EmaGramConfig, EmaGramState

from experiments.ema_gram.history_geometry import (
    combine_histories,
    frozen_history_step,
    pool_raw_histories,
    replace_matrix_history,
)
from experiments.ema_gram.probe import matrix_state


def test_all_matrix_histories_match_production_and_leave_nonmatrix_state_untouched():
    config = QwenConfig(hidden_dim=8, intermediate_dim=16, num_layers=2, num_heads=2, num_kv_heads=2, max_seq_len=8)
    model = config.build(hax.Axis("vocab", 32), key=jax.random.PRNGKey(17))
    optimizer_config = EmaGramConfig(learning_rate=0.001, warmup=0, alpha=0.125, ridge=0.01)
    optimizer = optimizer_config.build(100)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    cfg = SimpleNamespace(optimizer=optimizer_config, trainer=SimpleNamespace(compute_axis_mapping={}))
    gradient = jax.tree.map(lambda x: jnp.ones_like(x) * 0.03 if eqx.is_array(x) else x, model)
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(-1, 1), ("data", "model"))
    with jax.sharding.set_mesh(mesh):
        advance, history = frozen_history_step(cfg, model, state)
        history = advance(gradient, history)
        _, advanced = optimizer.update(gradient, state, eqx.filter(model, eqx.is_inexact_array))
    production = matrix_state(model, advanced).optimizer
    for actual, expected in zip(jax.tree.leaves(history), jax.tree.leaves(production), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
    fresh = jax.tree.map(lambda x: x * 2, history)
    mixed = combine_histories(production, fresh, momentum="fresh", grams="saved")
    replaced = replace_matrix_history(advanced, mixed)

    def without_matrix(x):
        return isinstance(x, EmaGramState)

    for before, after in zip(
        jax.tree.leaves(advanced, is_leaf=without_matrix), jax.tree.leaves(replaced, is_leaf=without_matrix), strict=True
    ):
        if not isinstance(before, EmaGramState):
            np.testing.assert_array_equal(before, after)
    for saved, candidate in zip(production.matrices, mixed.matrices, strict=True):
        np.testing.assert_allclose(candidate.momentum, 2 * saved.momentum)
        np.testing.assert_array_equal(candidate.left, saved.left)
        np.testing.assert_array_equal(candidate.right, saved.right)
    pooled = pool_raw_histories([history, fresh])
    for actual, raw in zip(jax.tree.leaves(pooled), jax.tree.leaves(history), strict=True):
        np.testing.assert_allclose(actual, 1.5 * raw, rtol=1e-6)
    assert float(optax.global_norm(history)) > 0
