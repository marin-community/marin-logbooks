# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from haliax.nn.loss import maybe_reduce_loss
from levanter.trainer_state import TrainerState

from experiments.post_training.identity_path_b.native import packed_example
from experiments.post_training.identity_path_b.recipe import (
    FROZEN_TOKEN_IDS,
    freeze_special_rows,
    loss_token_weights,
    restore_frozen_state,
    special_row_values,
)


class Vocabulary(eqx.Module):
    token_embed: jax.Array
    output_proj: jax.Array


def test_frozen_token_rows_survive_inherited_adam_momentum_and_decay():
    # Start with nonzero moments to prove that masking only gradients is insufficient.
    params = Vocabulary(jnp.ones((128256, 2)), jnp.ones((2, 128256)))
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    adam = optax.adamw(learning_rate=0.01, weight_decay=0.1)
    _, moments = adam.update(grads, adam.init(params), params)
    wrapped = freeze_special_rows(adam)
    updates, _ = wrapped.update(grads, moments, params)
    changed = optax.apply_updates(params, updates)
    ids = np.asarray(FROZEN_TOKEN_IDS)
    np.testing.assert_array_equal(changed.token_embed[ids], params.token_embed[ids])
    np.testing.assert_array_equal(changed.output_proj[:, ids], params.output_proj[:, ids])
    assert not np.array_equal(changed.token_embed[42], params.token_embed[42])
    assert not np.array_equal(changed.output_proj[:, 42], params.output_proj[:, 42])


def test_native_optimizer_step_restores_signed_zero_bits_and_progress():
    ids = jnp.asarray(FROZEN_TOKEN_IDS)
    params = Vocabulary(
        jnp.ones((128256, 2)).at[ids, 0].set(jnp.asarray(-0.0)),
        jnp.ones((2, 128256)).at[0, ids].set(jnp.asarray(-0.0)),
    )
    snapshots = special_row_values(params)
    optimizer = freeze_special_rows(optax.adamw(learning_rate=0.01, weight_decay=0.1))
    state = TrainerState.init(optimizer, params, key=jax.random.key(310))
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    for _ in range(2):
        state, _ = state.take_step(grads, key=jax.random.key(310))
        state = restore_frozen_state(state, snapshots)
    # Compare representation, not numeric equality: -0.0 == +0.0 numerically.
    expected = np.asarray(params.token_embed)[np.asarray(ids)].view(np.uint32)
    actual = np.asarray(state.model.token_embed)[np.asarray(ids)].view(np.uint32)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        np.asarray(state.model.output_proj)[:, np.asarray(ids)].view(np.uint32),
        np.asarray(params.output_proj)[:, np.asarray(ids)].view(np.uint32),
    )
    assert int(state.step) == 2
    assert not np.array_equal(state.model.token_embed[42], params.token_embed[42])


def test_packed_loss_follows_predicted_token_and_excludes_padding():
    tokens = jnp.asarray([128007, 128003, 42, 128009, 128001])
    weights = loss_token_weights(tokens, jnp.asarray([0, 1, 1, 1, 0]))
    example = packed_example(
        {
            "input_ids": tokens,
            "target_weights": weights,
            "segment_ids": [0, 0, 0, 0, -1],
        }
    )
    np.testing.assert_allclose(example.loss_weight, [0.1, 1, 1, 0, 0], rtol=0, atol=1e-8)
    assert example.attn_mask.segment_ids is not None
    query, keys = example.attn_mask.segment_ids
    np.testing.assert_array_equal(query, [0, 0, 0, 0, -1])
    np.testing.assert_array_equal(keys, query)


def test_all_semantic_controls_and_reserved_ids_use_end_exceptions():
    tokens = jnp.asarray([128000, 128001, 128004, 128008, 128009, 128010, 128255, 42])
    actual = loss_token_weights(tokens, jnp.ones_like(tokens))
    np.testing.assert_allclose(actual, [0.1, 1, 0.1, 0.1, 1, 0.1, 0.1, 1], rtol=0, atol=1e-8)


def test_native_weighted_loss_is_one_ratio_across_unequal_examples():
    batch, position = hax.Axis("batch", 2), hax.Axis("position", 3)
    losses = np.asarray([[2, 4, 8], [1, 3, 9]], dtype=np.float32)
    weights = np.asarray([[0.1, 1, 1], [1, 0, 0]], dtype=np.float32)
    actual = maybe_reduce_loss(
        hax.named(losses, (batch, position)),
        hax.mean,
        None,
        None,
        hax.named(weights, (batch, position)),
    )
    expected = np.sum(losses * weights) / np.sum(weights)
    np.testing.assert_allclose(actual.scalar(), expected, rtol=1e-6)
    assert not np.isclose(actual.scalar(), ((12.2 / 2.1) + 1) / 2)
