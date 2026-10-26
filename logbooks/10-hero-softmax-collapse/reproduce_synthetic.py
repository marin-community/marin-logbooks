# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce pure-CE collapse with the streaming kernel on CPU or GPU."""

import json

import jax
import jax.numpy as jnp
from levanter.kernels.pallas.fused_cross_entropy_loss.config import BlockSizes
from levanter.kernels.pallas.fused_cross_entropy_loss.xla import linear_softmax_cross_entropy_loss_xla


def main():
    hidden = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.bfloat16)
    targets = jnp.array([0], dtype=jnp.int32)
    for scores in ([9.0, -9.0, -9.0], [20.0, 6.0, 6.0]):
        head = jnp.zeros((3, 3), dtype=jnp.bfloat16).at[0].set(jnp.array(scores, dtype=jnp.bfloat16))

        def loss(weight):
            ce, _ = linear_softmax_cross_entropy_loss_xla(
                hidden,
                targets,
                weight,
                block_sizes=BlockSizes(b_block_size=1, v_block_size=2),
                dtype=jnp.float32,
                fast_backward=True,
            )
            return ce.sum()

        ce, gradient = jax.jit(jax.value_and_grad(loss))(head)
        logits = jnp.array(scores, dtype=jnp.float32)
        margin = logits[0] - jax.nn.logsumexp(logits[1:])
        print(
            json.dumps(
                {
                    "logits": scores,
                    "margin": float(margin),
                    "stable_ce": float(jax.nn.softplus(-margin)),
                    "streaming_ce": float(ce),
                    "head_first_row_gradient": gradient[0].astype(jnp.float32).tolist(),
                }
            )
        )


if __name__ == "__main__":
    main()
