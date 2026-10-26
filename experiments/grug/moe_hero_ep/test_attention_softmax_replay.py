# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.moe_hero_ep.attention_softmax_capture import AttentionCapture
from experiments.grug.moe_hero_ep.attention_softmax_replay import select_attention_panel


def test_attention_panel_keeps_full_keys_and_only_selected_real_cotangents():
    q = jnp.arange(2 * 8 * 8 * 4, dtype=jnp.float32).reshape(2, 8, 8, 4)
    k = jnp.arange(2 * 8 * 2 * 4, dtype=jnp.float32).reshape(2, 8, 2, 4)
    dout = q + 1
    bounds = jnp.array([[0, 0, 0, 0, 4, 4, 4, 4]] * 2)
    valid = jnp.ones((2, 8), dtype=jnp.bool_)
    capture = AttentionCapture(q, k, k + 1, q + 2, dout, bounds, valid, jnp.asarray(0.0), jnp.asarray(1.0))
    panel = jax.jit(lambda value: select_attention_panel(value, rank=jnp.asarray(1), layer_index=0, query_count=4))(
        capture
    )
    positions = np.asarray(panel.positions)
    np.testing.assert_array_equal(np.asarray(panel.q), np.asarray(q)[1:2, :, 4:8])
    np.testing.assert_array_equal(np.asarray(panel.k), np.asarray(k)[1:2, :, 1:2])
    np.testing.assert_array_equal(np.asarray(panel.v), np.asarray(k + 1)[1:2, :, 1:2])
    np.testing.assert_array_equal(np.asarray(panel.original_output), np.asarray(q + 2)[1:2, :, 4:8])
    assert np.all(positions // 2 == np.arange(4))
    assert np.all(positions > np.asarray(bounds)[1, positions])
    sparse = np.asarray(panel.sparse_dout)
    expected = np.zeros_like(sparse)
    expected[:, positions] = np.asarray(dout)[1:2, positions, 4:8]
    np.testing.assert_array_equal(sparse, expected)
    assert int(panel.selection_fallback_count) == 0


def test_attention_panel_exposes_single_key_and_invalid_strata():
    q = jnp.ones((1, 8, 4, 2))
    k = jnp.ones((1, 8, 1, 2))
    # The second stratum has no valid query; all other valid rows have one key.
    valid = jnp.array([[True, True, False, False, True, True, True, True]])
    bounds = jnp.arange(8)[None, :]
    capture = AttentionCapture(q, k, k, q, q * 3, bounds, valid, jnp.asarray(0.0), jnp.asarray(1.0))
    panel = select_attention_panel(capture, rank=jnp.asarray(0), layer_index=3, query_count=4)
    positions = np.asarray(panel.positions)
    assert int(panel.selection_fallback_count) == 4
    np.testing.assert_array_equal(np.asarray(panel.valid)[0, positions], [True, False, True, True])
    np.testing.assert_array_equal(np.asarray(panel.lower_bounds)[0, positions], positions)
    # The selected support is stable across repeat measurements/checkpoints.
    repeated = select_attention_panel(capture, rank=jnp.asarray(0), layer_index=3, query_count=4)
    np.testing.assert_array_equal(panel.positions, repeated.positions)
