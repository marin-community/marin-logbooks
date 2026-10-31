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
from jax.sharding import Mesh, NamedSharding
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


def parallel_eigh(normalized, *, mesh, axis_name, implementation):
    """Shard only full eigendecompositions, after the caller normalizes its Grams."""
    replicated = NamedSharding(mesh, P())
    normalized = jax.lax.with_sharding_constraint(normalized, replicated)
    normalized = jax.lax.optimization_barrier(normalized)
    axes = (axis_name,) if isinstance(axis_name, str) else axis_name
    shards = math.prod(mesh.shape[name] for name in axes)
    batch_shape = normalized.shape[:-2]
    count = math.prod(batch_shape)
    padded_count = math.ceil(count / shards) * shards
    flat = normalized.reshape((count, *normalized.shape[-2:]))
    padded = jnp.pad(flat, ((0, padded_count - count), (0, 0), (0, 0)))
    mapped = jax.shard_map(
        partial(jax.lax.linalg.eigh, implementation=implementation),
        mesh=mesh,
        axis_names=set(axes),
        in_specs=P(axis_name, None, None),
        out_specs=(P(axis_name, None, None), P(axis_name, None)),
    )
    vectors, values = jax.lax.optimization_barrier(mapped(padded))
    vectors = vectors[:count].reshape((*batch_shape, *normalized.shape[-2:]))
    values = values[:count].reshape((*batch_shape, normalized.shape[-1]))
    return jax.tree.map(lambda x: jax.lax.with_sharding_constraint(x, replicated), (vectors, values))


def eig_parallel_update(gradient, state, *, mesh, axis_name, alpha, ridge, eigh_implementation=None):
    state = update_gram_state(gradient, state, 0.95, 0.99)
    direction = gram_direction(
        state,
        alpha=alpha,
        ridge=ridge,
        polar_steps=5,
        eigh_implementation=eigh_implementation,
        eigh_fn=partial(parallel_eigh, mesh=mesh, axis_name=axis_name),
    )
    return direction, state
