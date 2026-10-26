# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.moe_hero_ep.attention_softmax_reference import (
    attention_reference,
    compare_attention_gradients,
)


def test_attention_reference_preserves_saturated_winner_gradient_and_reports_output_rounding():
    q = np.array([[[[1.0]]]])
    k = np.array([[[[10.0]], [[0.0]]]])
    v = np.array([[[[1.0]], [[0.0]]]])
    dout = np.ones_like(q)
    mask = np.ones((1, 1, 2), dtype=bool)
    reference = attention_reference(q, k, v, dout, mask, softmax_scale=1.0)
    expected = math.exp(-10) / (1 + math.exp(-10)) ** 2
    np.testing.assert_allclose(reference.dk[0, :, 0, 0], [expected, -expected], rtol=1e-12, atol=0)
    np.testing.assert_allclose(reference.dq[0, 0, 0, 0], 10 * expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose(reference.score_gradients.sum(), 0, atol=1e-20)

    # Rounded O=1 causes the winning score derivative to disappear while the
    # losing derivative remains negative, matching the fused backward formula.
    rounded_output = np.asarray(jnp.asarray(reference.output, dtype=jnp.bfloat16), dtype=np.float64)
    rounded_ds = reference.probabilities * np.array([0.0, -1.0])
    metrics = compare_attention_gradients(
        reference,
        rounded_output,
        np.zeros_like(q),
        rounded_ds[0, 0, 0, :, None, None].reshape(k.shape),
        q=q,
        k=k,
        v=v,
        dout=dout,
        softmax_scale=1.0,
    )
    assert metrics["rows"]["stored_output_zero_winner_nonzero_reference_count"] == 1
    assert metrics["dq"]["zero_actual_nonzero_reference_vectors"] == 1
    assert metrics["dq"]["relative_l2_error"] == 1
    event = metrics["worst_dq_events"][0]
    assert event["stored_output_reconstructed_winner_score_gradient"] == 0
    np.testing.assert_allclose(event["reference_winner_score_gradient"], expected, rtol=1e-12, atol=0)
    assert event["stored_output_max_absolute_score_gradient_error"] >= expected
    assert metrics["worst_cancellation_events"] == [event]
    assert metrics["largest_margin_events"] == [event]
    for control in metrics["controls"].values():
        assert control["zero_winner_nonzero_reference_count"] == 1
        assert control["dq"]["relative_l2_error"] == 1
        np.testing.assert_allclose(control["dq_fused_error_alignment"]["cosine_error_alignment"], 1, atol=1e-15)
        assert control["dq_fused_error_alignment"]["residual_error_l2_over_fused_error_l2"] == 0

    # Retain a cancellation event even when a different query dominates dQ error.
    two_queries = np.concatenate([q, q * 0.1], axis=1)
    two_dout = np.ones_like(two_queries)
    panel = attention_reference(two_queries, k, v, two_dout, np.ones((1, 2, 2), bool), softmax_scale=1.0)
    panel_output = panel.output.copy()
    panel_output[:, 0] = rounded_output[:, 0]
    panel_dq = panel.dq.copy()
    panel_dq[:, 1] += 100
    panel_metrics = compare_attention_gradients(
        panel,
        panel_output,
        panel_dq,
        panel.dk,
        q=two_queries,
        k=k,
        v=v,
        dout=two_dout,
        softmax_scale=1.0,
        event_count=1,
    )
    assert panel_metrics["worst_dq_events"][0]["query"] == 1
    assert panel_metrics["worst_cancellation_events"][0]["query"] == 0
    assert panel_metrics["largest_margin_events"][0]["query"] == 0


def test_attention_reference_remains_stable_beyond_fp64_probability_precision_cliff():
    q = np.array([[[[1.0]]]])
    k = np.array([[[[45.0]], [[0.0]]]])
    v = np.array([[[[1.0]], [[0.0]]]])
    reference = attention_reference(q, k, v, np.ones_like(q), np.ones((1, 1, 2), bool), softmax_scale=1.0)
    assert reference.pmax[0, 0, 0] == 1
    np.testing.assert_allclose(reference.dk[0, :, 0, 0], [math.exp(-45), -math.exp(-45)], rtol=1e-12, atol=0)
    np.testing.assert_allclose(reference.margin[0, 0, 0], 45, rtol=0, atol=0)


def test_attention_reference_gqa_and_masks_match_independent_fp64_autodiff():
    rng = np.random.default_rng(901)
    q = rng.normal(size=(1, 3, 4, 3)) * 0.4
    k = rng.normal(size=(1, 5, 2, 3)) * 0.4
    v = rng.normal(size=(1, 5, 2, 2))
    dout = rng.normal(size=(1, 3, 4, 2))
    # Selected original queries are positions 0, 3, 4. Respect one document
    # boundary at key2 and a sliding window of two keys at the final query.
    mask = np.array([[[1, 0, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]]], dtype=bool)
    scale = 1 / math.sqrt(3)
    reference = attention_reference(q, k, v, dout, mask, softmax_scale=scale)
    with jax.enable_x64():
        q_jax, k_jax, v_jax, do_jax = [jnp.array(value, dtype=jnp.float64) for value in (q, k, v, dout)]

        def loss(q_arg, k_arg, v_arg):
            expanded_k = jnp.repeat(k_arg, 2, axis=2)
            expanded_v = jnp.repeat(v_arg, 2, axis=2)
            scores = jnp.einsum("bqhd,bkhd->bqhk", q_arg, expanded_k) * scale
            probabilities = jax.nn.softmax(jnp.where(mask[:, :, None, :], scores, -jnp.inf), axis=-1)
            output = jnp.einsum("bqhk,bkhd->bqhd", probabilities, expanded_v)
            return jnp.sum(output * do_jax), output

        (_, output), gradients = jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)(q_jax, k_jax, v_jax)
    for actual, expected in zip((reference.dq, reference.dk, reference.dv), gradients, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(reference.output, output, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(reference.dq[:, 0], 0)
    np.testing.assert_array_equal(reference.dk[:, :2], 0)
    metrics = compare_attention_gradients(
        reference,
        output,
        gradients[0],
        gradients[1],
        q=q,
        k=k,
        v=v,
        dout=dout,
        softmax_scale=scale,
        fused_dv=gradients[2],
    )
    assert metrics["rows"]["single_key_count"] == 4
    assert metrics["rows"]["multi_key_count"] == 8
    assert sum(metrics["rows"]["margin_histogram"]) == 8
    assert metrics["dq"]["relative_l2_error"] < 1e-12
    assert metrics["dk"]["relative_l2_error"] < 1e-12


def test_attention_reference_empty_query_has_no_output_or_gradient():
    q = np.ones((1, 1, 2, 2))
    k = np.ones((1, 3, 1, 2))
    v = np.arange(6).reshape(1, 3, 1, 2)
    reference = attention_reference(q, k, v, q, np.zeros((1, 1, 3), dtype=bool), softmax_scale=1.0)
    for value in (reference.output, reference.dq, reference.dk, reference.dv):
        np.testing.assert_array_equal(value, 0)
    metrics = compare_attention_gradients(
        reference,
        reference.output,
        reference.dq,
        reference.dk,
        q=q,
        k=k,
        v=v,
        dout=q,
        softmax_scale=1.0,
        fused_dv=reference.dv,
    )
    assert metrics["rows"]["masked_count"] == 2
    assert metrics["rows"]["multi_key_count"] == 0
    assert metrics["worst_dq_events"] == []
    assert metrics["worst_cancellation_events"] == []
    assert metrics["largest_margin_events"] == []
