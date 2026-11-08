# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent tiny NumPy oracle for the new full-model SVD loss-control direction."""

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from levanter.models.qwen import Qwen3Config
from levanter.optim.ema_gram import NEWTON_SCHULZ, SVD, EmaGramConfig, EmaGramState, GramState
from levanter.optim.util import flatten_linear_layers
from levanter.trainer_state import TrainerState
from probe_svd_check import numpy_polar, numpy_power
from supplemental_probe import matched_direction, summarize_losses

from experiments.ema_gram.probe import matrix_state, norm_matched_direction


def execute():
    cfg = Qwen3Config(
        hidden_dim=8, intermediate_dim=16, num_layers=2, num_heads=2, num_kv_heads=2, head_dim=4, max_seq_len=8
    )
    model = cfg.build(hax.Axis("vocab", 16), key=jax.random.PRNGKey(1))
    state = TrainerState.init(EmaGramConfig(alpha=0.25).build(20), model, key=jax.random.PRNGKey(2))
    optimizer = matrix_state(state.model, state.opt_state).optimizer
    rng = np.random.default_rng(487)
    banks = []
    for bank in optimizer.matrices:
        grams = []
        for shape in (bank.left.shape, bank.right.shape):
            a = rng.normal(size=shape).astype(np.float32)
            grams.append(jnp.asarray(a @ a.swapaxes(-1, -2) + np.eye(shape[-1])))
        banks.append(GramState(jnp.asarray(rng.normal(size=bank.momentum.shape), dtype=jnp.float32), *grams))
    opt_state = jax.tree.map(
        lambda x: EmaGramState(tuple(banks)) if isinstance(x, EmaGramState) else x,
        state.opt_state,
        is_leaf=lambda x: isinstance(x, EmaGramState),
    )
    before = [np.asarray(x).copy() for x in jax.tree.leaves(opt_state)]
    production = matched_direction(state.model, opt_state, 0.25, 0.01, NEWTON_SCHULZ, 5)
    prior = norm_matched_direction(state.model, opt_state, alpha=0.25, ridge=0.01, polar_steps=5)
    for got, expected in zip(jax.tree.leaves(production), jax.tree.leaves(prior), strict=True):
        np.testing.assert_allclose(got, expected, rtol=2e-4, atol=2e-5)
    reference = matched_direction(state.model, opt_state, 0.0, 0.01, NEWTON_SCHULZ, 5)
    exact = matched_direction(state.model, opt_state, 0.25, 0.01, SVD, 5)
    labels = matrix_state(state.model, opt_state)

    # Reapply the same model mask to align flattened matrices with the stored bank order.
    def masked_leaves(direction):
        masked = jax.tree.map(
            lambda mask, d: mask if isinstance(mask, optax.MaskedNode) else d,
            labels.masked_model,
            direction,
            is_leaf=lambda x: isinstance(x, optax.MaskedNode),
        )
        return jax.tree.leaves(flatten_linear_layers(masked))

    count = 0
    for bank, got, target in zip(banks, masked_leaves(exact), masked_leaves(reference), strict=True):
        for m, l, r, actual, ref in zip(
            *(np.asarray(x).reshape((-1, *x.shape[-2:])) for x in (*bank, got, target)), strict=True
        ):
            left, right = numpy_power(l, 0.25, 0.01), numpy_power(r, 0.25, 0.01)
            expected = left @ numpy_polar(left @ m @ right) @ right
            expected *= np.linalg.norm(ref) / np.linalg.norm(expected)
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
            count += 1
    np.testing.assert_array_equal(exact.embeddings.token_embeddings.weight.array, 0)
    np.testing.assert_array_equal(exact.lm_head.weight.array, 0)
    for old, current in zip(before, jax.tree.leaves(opt_state), strict=True):
        np.testing.assert_array_equal(old, current)
    summary = summarize_losses([2.0, 4.0, 6.0, 8.0], [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(
        [summary["delta_mean"], summary["delta_std"], summary["delta_sem"]], [4.0, np.sqrt(20 / 3), np.sqrt(20 / 3) / 2]
    )
    return dict(
        svd_matrices_checked=count,
        production_parity=True,
        nonmatrices_fixed=True,
        state_unchanged=True,
        paired_summary_oracle=True,
    )
