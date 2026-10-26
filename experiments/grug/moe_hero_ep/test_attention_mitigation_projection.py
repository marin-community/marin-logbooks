# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.moe_hero_ep.attention_mitigation_projection import (
    project_native_attention_operands,
    projected_attention_backward,
    projection_reference,
)
from experiments.grug.moe_hero_ep.attention_softmax_reference import attention_reference


def test_projection_uses_consumed_probability_mass_and_corrects_both_gradients():
    # Nonunit BF16 probability mass makes using rho instead of rho/m wrong.
    q = np.array([[[[2.0, -1.0]]]])
    k = np.array([[[[16.0, 1.0]], [[16.0, 3.0]], [[16.0, 5.0]]]])
    probability = np.array([[[[0.5, 0.25, 0.125]]]])
    score = np.array([[[[1.0, -0.25, 0.125]]]])
    mask = np.ones((1, 1, 3), dtype=bool)
    reference = projection_reference(q, k, score, probability, mask, softmax_scale=0.5)
    np.testing.assert_allclose(reference.projected_scores.sum(axis=-1), 0, atol=2e-16)
    np.testing.assert_allclose(reference.dq[..., 0], 0, atol=2e-15)
    np.testing.assert_allclose(reference.dk.sum(axis=1), 0, atol=2e-15)
    translated = projection_reference(q, k + 1024, score, probability, mask, softmax_scale=0.5)
    np.testing.assert_allclose(reference.dq, translated.dq, atol=2e-13)
    assert abs((score - reference.row_mass[..., None] * probability).sum()) > 0.1


def test_postcast_projection_removes_paper_four_key_translation_leak():
    # Paper's cast witness: (1+eta,-1,-eta,0) becomes (1,-1,-eta,0).
    eta = 3 / 1024
    q = jnp.zeros((1, 1, 1, 2), dtype=jnp.bfloat16)
    k = jnp.asarray([[[[65536.0, float(index)]] for index in range(4)]], dtype=jnp.bfloat16)
    probabilities = jnp.full((1, 1, 1, 4), 0.25, dtype=jnp.bfloat16)
    score = jnp.asarray([[[[1 + eta, -1.0, -eta, 0.0]]]], dtype=jnp.bfloat16)
    raw_dq = jnp.einsum("bqhk,bkhd->bqhd", score, k, preferred_element_type=jnp.float32)
    raw_dk = jnp.zeros_like(k, dtype=jnp.float32)
    projected = jax.jit(project_native_attention_operands, static_argnames=("softmax_scale",))(
        q, k, probabilities, score, raw_dq, raw_dk, softmax_scale=0.125
    )
    assert float(raw_dq[0, 0, 0, 0] * 0.125) == -24
    assert float(projected.dq[0, 0, 0, 0]) == 0
    assert float(projected.row_mass[0, 0, 0]) == -eta
    assert float(projected.post_projection_mass[0, 0, 0]) == 0


def test_projected_backward_recovers_two_key_saved_output_cancellation():
    q = jnp.asarray([[[[1.0, 0.0]]]], dtype=jnp.bfloat16)
    k = jnp.asarray([[[[10.0, 4.0]], [[0.0, -4.0]]]], dtype=jnp.bfloat16)
    v = jnp.asarray([[[[1.0]], [[0.0]]]], dtype=jnp.bfloat16)
    dout = jnp.ones((1, 1, 1, 1), dtype=jnp.bfloat16)
    stored = jnp.ones_like(dout)
    mask = jnp.ones((1, 1, 2), dtype=bool)
    reference = attention_reference(q, k, v, dout, mask, softmax_scale=1.0)
    actual = jax.jit(projected_attention_backward, static_argnames=("softmax_scale",))(
        q, k, v, dout, stored, mask, softmax_scale=1.0
    )
    assert float(actual.unprojected_dq[0, 0, 0, 0]) == 0
    # Independent two-key scalar arithmetic contract: BF16 P and dS, FP32
    # contractions, BF16(lambda*Q) only in the key correction, final BF16 cast.
    logits = np.array([10.0, 0.0], dtype=np.float32)
    probability = np.exp(logits - np.logaddexp(logits[0], logits[1]))
    r0, r1 = np.asarray(jnp.asarray(probability, dtype=jnp.bfloat16), dtype=np.float32)
    t0, t1 = np.asarray(jnp.asarray(probability * np.array([0.0, -1.0]), dtype=jnp.bfloat16), dtype=np.float32)
    coefficient = np.float32((t0 + t1) / (r0 + r1))
    raw_q = np.array([10 * t0, 4 * t0 - 4 * t1], dtype=np.float32)
    probability_keys = np.array([10 * r0, 4 * r0 - 4 * r1], dtype=np.float32)
    expected_q = jnp.asarray(raw_q - coefficient * probability_keys, dtype=jnp.bfloat16)
    rounded_coefficient = np.float32(jnp.asarray(coefficient, dtype=jnp.bfloat16))
    expected_k = jnp.asarray(
        [[t0 - r0 * rounded_coefficient, 0], [t1 - r1 * rounded_coefficient, 0]], dtype=jnp.bfloat16
    )
    np.testing.assert_array_equal(actual.dq.reshape(2), expected_q)
    np.testing.assert_array_equal(actual.dk.reshape(2, 2), expected_k)
    baseline_error = np.linalg.norm(np.asarray(actual.unprojected_dq, dtype=np.float32) - reference.dq)
    projected_error = np.linalg.norm(np.asarray(actual.dq, dtype=np.float32) - reference.dq)
    assert projected_error < baseline_error / 100


def test_native_operand_projection_matches_masked_grouped_fp64_operator():
    # Dyadic operands keep correction*Q exactly representable in BF16, isolating
    # the Q/K operator and GQA aggregation from the key-pass extra rounding.
    q = jnp.asarray(np.arange(32).reshape(1, 4, 4, 2) / 8, dtype=jnp.bfloat16)
    k = jnp.asarray(np.arange(20).reshape(1, 5, 2, 2) / 4, dtype=jnp.bfloat16)
    mask = np.array(
        [
            [
                [True, True, False, False, False],
                [False, True, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, False],
            ]
        ]
    )
    probabilities = np.broadcast_to(mask[:, :, None, :] * 0.5, (1, 4, 4, 5)).copy()
    score = probabilities * np.array([1, -0.5, 0.25, 0.125, 0])
    probability_operand = jnp.asarray(probabilities, dtype=jnp.bfloat16)
    score_operand = jnp.asarray(score, dtype=jnp.bfloat16)
    raw_dq = jnp.einsum(
        "bqhrk,bkhd->bqhrd", score_operand.reshape(1, 4, 2, 2, 5), k, preferred_element_type=jnp.float32
    ).reshape(q.shape)
    raw_dk = jnp.einsum(
        "bqhrk,bqhrd->bkhd",
        score_operand.reshape(1, 4, 2, 2, 5),
        q.reshape(1, 4, 2, 2, 2),
        preferred_element_type=jnp.float32,
    )
    reference = projection_reference(q, k, score_operand, probability_operand, mask, softmax_scale=0.5)
    actual = project_native_attention_operands(
        q, k, probability_operand, score_operand, raw_dq, raw_dk, softmax_scale=0.5
    )
    np.testing.assert_array_equal(np.asarray(actual.dq, dtype=np.float32), reference.dq)
    np.testing.assert_array_equal(np.asarray(actual.dk, dtype=np.float32), reference.dk)
    np.testing.assert_array_equal(actual.post_projection_mass, 0)
    np.testing.assert_array_equal(actual.dq[:, 2:], 0)
    np.testing.assert_array_equal(actual.dk[:, 4:], 0)
