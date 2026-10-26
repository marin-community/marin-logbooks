# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage of mixed-precision correction feeding native backward."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.moe_hero_ep.attention_mitigation_saved_output import saved_output_preprocess


def test_saved_output_preprocess_preserves_correction_lost_by_bf16_and_pads_native_rows():
    # Public BF16 O rounds both nonzero components to one, losing the entire
    # correction. The pre-cast FP32 state must preserve this genuine difference.
    output = np.zeros((1, 3, 4, 128), dtype=np.float32)
    output[..., 0] = 1 - 2**-14
    output[..., 1] = 1 + 2**-13
    dout = np.zeros_like(output)
    dout[..., 0], dout[..., 1] = 1, -1
    lse = np.arange(12, dtype=np.float32).reshape(1, 4, 3) / 4
    lse[0, 2, 1] = -np.inf

    correction, lse_log2 = jax.jit(partial(saved_output_preprocess, tile_m=128))(
        jnp.asarray(output), jnp.asarray(dout, dtype=jnp.bfloat16), jnp.asarray(lse)
    )
    reference = np.sum(output.astype(np.float64) * dout.astype(np.float64), axis=-1).transpose(0, 2, 1)
    rounded = np.asarray(jnp.asarray(output, dtype=jnp.bfloat16), dtype=np.float64)
    rounded_correction = np.sum(rounded * dout, axis=-1)
    np.testing.assert_array_equal(rounded_correction, np.zeros((1, 3, 4)))
    np.testing.assert_array_equal(np.asarray(correction)[..., :3], reference)
    np.testing.assert_array_equal(np.asarray(correction)[..., 3:], np.zeros((1, 4, 125)))
    expected_log2 = np.where(np.isneginf(lse), 0, lse.astype(np.float64) / np.log(2))
    np.testing.assert_allclose(np.asarray(lse_log2)[..., :3], expected_log2, rtol=1e-6, atol=0)
    np.testing.assert_array_equal(np.asarray(lse_log2)[..., 3:], np.zeros((1, 4, 125)))
