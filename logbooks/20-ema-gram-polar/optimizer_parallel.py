# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Research candidate: assign independent, full Gram updates to separate chips.

Padding only adds independent zero matrices along the layer axis. Each real
matrix retains its full left/right Grams and the production update equations.
This module is not selected by the training configuration or the queued job.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from levanter.optim.ema_gram import GramState, gram_direction, update_gram_state


def local_update(gradient, state, *, alpha, ridge, eigh_implementation):
    state = update_gram_state(gradient, state, 0.95, 0.99)
    direction = gram_direction(state, alpha=alpha, ridge=ridge, polar_steps=5, eigh_implementation=eigh_implementation)
    return direction, state


def parallel_update(
    gradient: jax.Array,
    state: GramState,
    *,
    mesh: Mesh,
    axis_name: str,
    alpha: float,
    ridge: float,
    eigh_implementation: jax.lax.linalg.EighImplementation | None = None,
) -> tuple[jax.Array, GramState]:
    """Distribute complete independent matrices, preserving the caller's shapes."""
    batch_shape = gradient.shape[:-2]
    count = math.prod(batch_shape)
    padded_count = math.ceil(count / mesh.shape[axis_name]) * mesh.shape[axis_name]

    def pad(array):
        flat = array.reshape((count, *array.shape[-2:]))
        return jnp.pad(flat, ((0, padded_count - count), (0, 0), (0, 0)))

    def unpad(array):
        return array[:count].reshape((*batch_shape, *array.shape[-2:]))

    matrix_spec = P(axis_name, None, None)
    state_spec = GramState(matrix_spec, matrix_spec, matrix_spec)
    mapped = jax.shard_map(
        partial(local_update, alpha=alpha, ridge=ridge, eigh_implementation=eigh_implementation),
        mesh=mesh,
        axis_names={axis_name},
        in_specs=(matrix_spec, state_spec),
        out_specs=(matrix_spec, state_spec),
    )
    output = mapped(pad(gradient), jax.tree.map(pad, state))
    return jax.tree.map(unpad, output)
