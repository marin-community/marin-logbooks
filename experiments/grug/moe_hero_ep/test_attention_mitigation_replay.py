# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.moe_hero_ep.attention_mitigation_replay import native_operand_export_checks, select_full_head_panel
from experiments.grug.moe_hero_ep.attention_softmax_capture import AttentionCapture
from experiments.grug.moe_hero_ep.attention_softmax_replay import select_attention_panel


def test_full_head_selection_preserves_prior_sequence_positions_and_every_head_cotangent():
    q = jnp.arange(3 * 16 * 8 * 4, dtype=jnp.float32).reshape(3, 16, 8, 4)
    k = jnp.arange(3 * 16 * 2 * 4, dtype=jnp.float32).reshape(3, 16, 2, 4)
    bounds = jnp.broadcast_to((jnp.arange(16) // 8) * 8, (3, 16))
    valid = jnp.ones((3, 16), dtype=jnp.bool_)
    capture = AttentionCapture(q, k, k + 1, q + 2, q + 3, bounds, valid, jnp.asarray(0.0), jnp.asarray(1.0))
    rank = jnp.asarray(2)
    selected = jax.jit(lambda value: select_full_head_panel(value, rank=rank, layer_index=3, query_count=4))(capture)
    prior = select_attention_panel(capture, rank=rank, layer_index=3, query_count=4)
    np.testing.assert_array_equal(selected.positions, prior.positions)
    assert int(selected.local_row) == int(prior.local_row) == 2
    np.testing.assert_array_equal(selected.q, np.asarray(q)[2:3])
    np.testing.assert_array_equal(selected.k, np.asarray(k)[2:3])
    np.testing.assert_array_equal(selected.v, np.asarray(k + 1)[2:3])
    np.testing.assert_array_equal(selected.original_output, np.asarray(q + 2)[2:3])
    expected = np.zeros((1, 16, 8, 4), dtype=np.float32)
    positions = np.asarray(selected.positions)
    expected[:, positions] = np.asarray(q + 3)[2:3, positions]
    np.testing.assert_array_equal(selected.dout, expected)
    assert np.count_nonzero(np.asarray(selected.dout)) == 4 * 8 * 4
    np.testing.assert_array_equal(selected.lower_bounds, np.asarray(bounds)[2:3])
    assert int(selected.selection_fallback_count) == 0


def test_full_head_selection_keeps_invalid_and_single_key_fallbacks_explicit():
    q = jnp.ones((1, 8, 8, 4))
    k = jnp.ones((1, 8, 2, 4))
    valid = jnp.array([[True, True, False, False, True, True, True, True]])
    bounds = jnp.arange(8)[None]
    capture = AttentionCapture(q, k, k, q, q * 3, bounds, valid, jnp.asarray(0.0), jnp.asarray(1.0))
    selected = select_full_head_panel(capture, rank=jnp.asarray(0), layer_index=0, query_count=4)
    assert int(selected.selection_fallback_count) == 4
    np.testing.assert_array_equal(
        np.asarray(selected.valid)[:, np.asarray(selected.positions)], [[True, False, True, True]]
    )
    np.testing.assert_array_equal(
        np.asarray(selected.lower_bounds)[:, np.asarray(selected.positions)], selected.positions[None]
    )
    assert selected.q.shape[2] == 8
    assert selected.k.shape[2] == 2


def test_native_export_checks_accept_quantized_operator_and_reject_missing_or_corrupt_exports():
    q = jnp.asarray(np.arange(16).reshape(1, 2, 4, 2) / 8, dtype=jnp.bfloat16)
    k = jnp.asarray(np.arange(8).reshape(1, 4, 1, 2) / 4, dtype=jnp.bfloat16)
    probability = jnp.broadcast_to(jnp.asarray([[[[0.5, 0.5, 0, 0]], [[0, 0, 0.25, 0.75]]]]), (1, 2, 4, 4))
    probability = probability.astype(jnp.bfloat16)
    score = probability * jnp.asarray([1, -0.5, 0.25, -0.125], dtype=jnp.bfloat16)
    expected_q = np.zeros(q.shape, dtype=np.float32)
    expected_k = np.zeros(k.shape, dtype=np.float32)
    for query in range(2):
        for head in range(4):
            for key in range(4):
                weight = float(score[0, query, head, key])
                expected_q[0, query, head] += weight * np.asarray(k[0, key, 0], dtype=np.float32)
                expected_k[0, key, 0] += weight * np.asarray(q[0, query, head], dtype=np.float32)
    positions = jnp.asarray([[1, 3]])
    lower = jnp.asarray([[0, 2]])
    valid = jnp.ones((1, 2), dtype=jnp.bool_)

    def checks(exported_probability, exported_score):
        return native_operand_export_checks(
            q,
            k,
            exported_probability,
            exported_score,
            jnp.asarray(expected_q),
            jnp.asarray(expected_k),
            positions,
            lower,
            valid,
            full_query_length=4,
        )

    accepted = jax.jit(checks)(probability, score)
    assert all(float(value) == 0 for value in accepted.values())
    missing = checks(jnp.zeros_like(probability), score)
    assert float(missing["export_invalid_probability_row_count"]) == 8
    outside = checks(probability.at[0, 0, 0, 3].set(0.5), score)
    assert float(outside["export_masked_nonzero_count"]) == 1
    corrupt = checks(probability, score.at[0, 0, 0, 0].set(1))
    assert float(corrupt["export_q_contraction_rounding_bound_excess"]) > 0
    assert float(corrupt["export_k_contraction_rounding_bound_excess"]) > 0
    nonfinite = checks(probability.at[0, 0, 0, 0].set(jnp.inf), score.at[0, 0, 0, 0].set(jnp.nan))
    assert float(nonfinite["export_invalid_probability_row_count"]) == 1
    assert float(nonfinite["export_nonfinite_score_count"]) == 1
