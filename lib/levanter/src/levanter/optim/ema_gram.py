# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-sided EMA gradient-Gram preconditioning of the momentum polar factor."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from levanter.optim.config import OptimizerConfig
from levanter.optim.muon import MuonConfig
from levanter.optim.util import flatten_linear_layers, unflatten_linear_layers, zeropower_via_newtonschulz5


NEWTON_SCHULZ = "newton_schulz"
SVD = "svd"


class GramState(NamedTuple):
    momentum: jax.Array
    left: jax.Array
    right: jax.Array


class EmaGramState(NamedTuple):
    matrices: tuple[GramState, ...]


def init_gram_state(matrix: jax.Array) -> GramState:
    """Allocate FP32 buffers; leading dimensions index independent matrices."""
    batch, rows, cols = matrix.shape[:-2], matrix.shape[-2], matrix.shape[-1]
    return GramState(
        jnp.zeros(matrix.shape, jnp.float32),
        jnp.zeros((*batch, rows, rows), jnp.float32),
        jnp.zeros((*batch, cols, cols), jnp.float32),
    )


@jax.named_call
def update_gram_state(gradient: jax.Array, state: GramState, beta1: float, beta2: float) -> GramState:
    """Accumulate the already globally reduced, clipped gradient before taking powers."""
    g = gradient.astype(jnp.float32)
    gt = jnp.swapaxes(g, -1, -2)
    return GramState(
        beta1 * state.momentum + (1 - beta1) * g,
        beta2 * state.left + (1 - beta2) * jnp.matmul(g, gt, precision=jax.lax.Precision.HIGHEST),
        beta2 * state.right + (1 - beta2) * jnp.matmul(gt, g, precision=jax.lax.Precision.HIGHEST),
    )


@jax.named_call
def relative_gram_power(
    gram: jax.Array,
    alpha: float,
    ridge: float,
    *,
    eigh_implementation: jax.lax.linalg.EighImplementation | None = None,
    eigh_fn: Callable[..., tuple[jax.Array, jax.Array]] = jax.lax.linalg.eigh,
    power_fn: Callable[..., jax.Array] | None = None,
) -> jax.Array:
    """Return (Gram / mean eigenvalue + ridge I)^(-alpha), or I for a zero Gram."""
    gram = (gram + jnp.swapaxes(gram, -1, -2)) * 0.5
    mean = jnp.trace(gram, axis1=-2, axis2=-1) / gram.shape[-1]
    normalized = gram / jnp.where(mean > 0, mean, 1)[..., None, None]
    if power_fn is None:
        vectors, values = eigh_fn(normalized, implementation=eigh_implementation)
        powers = (jnp.maximum(values, 0) + ridge) ** (-alpha)
        power = jnp.matmul(
            vectors * powers[..., None, :], jnp.swapaxes(vectors, -1, -2), precision=jax.lax.Precision.HIGHEST
        )
    else:
        power = power_fn(normalized, alpha=alpha, ridge=ridge, implementation=eigh_implementation)
    return jnp.where((mean > 0)[..., None, None], power, jnp.eye(gram.shape[-1], dtype=gram.dtype))


def matrix_parallel_eigh(
    normalized: jax.Array,
    *,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_name: str | tuple[str, ...],
    implementation: jax.lax.linalg.EighImplementation | None,
) -> tuple[jax.Array, jax.Array]:
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
        # TPU QDWH initializes a convergence flag as invariant, then updates it
        # from shard-local data. JAX 0.11.1 rejects that internal loop under VMA.
        # Every manual axis appears in both output specs: no replicated-output
        # promise is omitted, and the numerical reference checks still apply.
        check_vma=False,
    )
    vectors, values = jax.lax.optimization_barrier(mapped(padded))
    vectors = vectors[:count].reshape((*batch_shape, *normalized.shape[-2:]))
    values = values[:count].reshape((*batch_shape, normalized.shape[-1]))
    return jax.tree.map(lambda x: jax.lax.with_sharding_constraint(x, replicated), (vectors, values))


def exact_polar(matrix: jax.Array) -> jax.Array:
    """SVD partial isometry with numerical zero modes removed (offline reference)."""
    u, singular, vh = jnp.linalg.svd(matrix, full_matrices=False)
    threshold = jnp.finfo(matrix.dtype).eps * max(matrix.shape[-2:]) * singular[..., :1]
    return jnp.matmul(u * (singular > threshold)[..., None, :], vh, precision=jax.lax.Precision.HIGHEST)


def matrix_parallel_power(
    normalized: jax.Array,
    *,
    alpha: float,
    ridge: float,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_name: str | tuple[str, ...],
    implementation: jax.lax.linalg.EighImplementation | None,
) -> jax.Array:
    """Reconstruct each full matrix power on its eigensolver shard before gathering."""
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

    def reconstruct(local):
        vectors, values = jax.lax.linalg.eigh(local, implementation=implementation)
        powers = (jnp.maximum(values, 0) + ridge) ** (-alpha)
        return jnp.matmul(
            vectors * powers[..., None, :], jnp.swapaxes(vectors, -1, -2), precision=jax.lax.Precision.HIGHEST
        )

    mapped = jax.shard_map(
        reconstruct,
        mesh=mesh,
        axis_names=set(axes),
        in_specs=P(axis_name, None, None),
        out_specs=P(axis_name, None, None),
        # Same scoped TPU QDWH VMA exception as matrix_parallel_eigh; all
        # manual axes remain present in the output specification.
        check_vma=False,
    )
    power = jax.lax.optimization_barrier(mapped(padded))
    power = power[:count].reshape((*batch_shape, *normalized.shape[-2:]))
    return jax.lax.with_sharding_constraint(power, replicated)


def grouped_gram_powers(
    grams: tuple[jax.Array, ...],
    *,
    alpha: float,
    ridge: float,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_name: str | tuple[str, ...],
    implementation: jax.lax.linalg.EighImplementation | None = None,
) -> tuple[jax.Array, ...]:
    """Batch independent full Grams of equal size before matrix-parallel eigh."""
    groups = {}
    means = []
    for index, gram in enumerate(grams):
        symmetric = (gram + jnp.swapaxes(gram, -1, -2)) * 0.5
        mean = jnp.trace(symmetric, axis1=-2, axis2=-1) / gram.shape[-1]
        normalized = symmetric / jnp.where(mean > 0, mean, 1)[..., None, None]
        means.append(mean)
        # Shape and dtype are static: Python only plans independent matrix banks.
        groups.setdefault((gram.shape[-1], gram.dtype), []).append((index, normalized))
    outputs = {}
    for (size, _), entries in groups.items():
        flat = jnp.concatenate([g.reshape((-1, size, size)) for _, g in entries], axis=0)
        vectors, values = matrix_parallel_eigh(flat, mesh=mesh, axis_name=axis_name, implementation=implementation)
        offset = 0
        for index, normalized in entries:
            count = math.prod(normalized.shape[:-2])
            v = vectors[offset : offset + count].reshape(normalized.shape)
            w = values[offset : offset + count].reshape(normalized.shape[:-1])
            weights = (jnp.maximum(w, 0) + ridge) ** (-alpha)
            power = jnp.matmul(v * weights[..., None, :], jnp.swapaxes(v, -1, -2), precision=jax.lax.Precision.HIGHEST)
            outputs[index] = jnp.where(
                (means[index] > 0)[..., None, None], power, jnp.eye(size, dtype=normalized.dtype)
            )
            offset += count
    return tuple(outputs[index] for index in range(len(grams)))


@jax.named_call
def gram_direction(
    state: GramState,
    *,
    alpha: float,
    ridge: float,
    polar_steps: int,
    polar_method: str = NEWTON_SCHULZ,
    eigh_implementation: jax.lax.linalg.EighImplementation | None = None,
    eigh_fn: Callable[..., tuple[jax.Array, jax.Array]] = jax.lax.linalg.eigh,
    power_fn: Callable[..., jax.Array] | None = None,
    precomputed_powers: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """Return a positive descent direction without advancing the supplied EMA state."""
    m = state.momentum
    if alpha:
        if precomputed_powers is None:
            left = relative_gram_power(
                state.left, alpha, ridge, eigh_implementation=eigh_implementation, eigh_fn=eigh_fn, power_fn=power_fn
            )
            right = relative_gram_power(
                state.right, alpha, ridge, eigh_implementation=eigh_implementation, eigh_fn=eigh_fn, power_fn=power_fn
            )
        else:
            left, right = precomputed_powers
        inner = jnp.matmul(
            jnp.matmul(left, m, precision=jax.lax.Precision.HIGHEST), right, precision=jax.lax.Precision.HIGHEST
        )
    else:
        inner = m
    if polar_method == SVD:
        polar = exact_polar(inner)
    elif polar_method == NEWTON_SCHULZ:
        flat = inner.reshape((-1, *inner.shape[-2:]))
        polar = jax.vmap(partial(zeropower_via_newtonschulz5, steps=polar_steps))(flat).reshape(inner.shape)
    else:
        raise ValueError(f"Unknown polar method: {polar_method}")
    if alpha:
        return jnp.matmul(
            jnp.matmul(left, polar, precision=jax.lax.Precision.HIGHEST), right, precision=jax.lax.Precision.HIGHEST
        )
    return polar


def scale_by_ema_gram(
    *,
    alpha: float,
    ridge: float,
    beta1: float = 0.95,
    beta2: float = 0.99,
    polar_steps: int = 5,
    eigh_layout: str = "replicated",
    eigh_implementation: str = "default",
) -> optax.GradientTransformation:
    """Transform matrix-only trees, preserving Haliax Linear layouts and scan axes.

    Every step refreshes both full (unblocked) Grams, including for alpha=0.
    Call after gradient accumulation/reduction and global clipping. No Nesterov,
    bias correction, aspect-ratio scaling, or update-norm constraint is applied.
    """
    if alpha < 0 or ridge <= 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or polar_steps < 1:
        raise ValueError("Invalid EMA Gram hyperparameters")

    if eigh_layout not in ("replicated", "matrix", "matrix_power", "matrix_grouped"):
        raise ValueError(f"Unknown eigensolver layout: {eigh_layout}")
    if eigh_implementation not in ("default", "jacobi"):
        raise ValueError(f"Unknown eigensolver implementation: {eigh_implementation}")
    implementation = None if eigh_implementation == "default" else jax.lax.linalg.EighImplementation.JACOBI

    def init_fn(params):
        leaves = jax.tree.leaves(flatten_linear_layers(params))
        if any(p.ndim < 2 for p in leaves):
            raise ValueError("EMA Gram accepts matrix parameters only; route vectors to Adam")
        return EmaGramState(tuple(init_gram_state(p) for p in leaves))

    def update_fn(updates, state, params=None):
        del params  # Optax permits callers to pass this keyword.
        leaves, treedef = jax.tree.flatten(flatten_linear_layers(updates))
        states = tuple(update_gram_state(g, s, beta1, beta2) for g, s in zip(leaves, state.matrices, strict=True))
        eig_fn = jax.lax.linalg.eigh
        power_fn = None
        powers: list[tuple[jax.Array, jax.Array] | None] = [None] * len(states)
        if eigh_layout in ("matrix", "matrix_power", "matrix_grouped") and alpha:
            mesh = jax.sharding.get_abstract_mesh()
            if mesh.empty:
                raise ValueError("Matrix-parallel EMA-Gram requires an active mesh")
            axes = tuple(name for name, size in mesh.shape.items() if size > 1)
            if axes:
                if eigh_layout == "matrix":
                    eig_fn = partial(matrix_parallel_eigh, mesh=mesh, axis_name=axes)
                elif eigh_layout == "matrix_power":
                    power_fn = partial(matrix_parallel_power, mesh=mesh, axis_name=axes)
                else:
                    grouped = grouped_gram_powers(
                        tuple(g for state in states for g in (state.left, state.right)),
                        alpha=alpha,
                        ridge=ridge,
                        mesh=mesh,
                        axis_name=axes,
                        implementation=implementation,
                    )
                    powers = list(zip(grouped[::2], grouped[1::2], strict=True))
        directions = [
            gram_direction(
                s,
                alpha=alpha,
                ridge=ridge,
                polar_steps=polar_steps,
                eigh_fn=eig_fn,
                power_fn=power_fn,
                eigh_implementation=implementation,
                precomputed_powers=p,
            ).astype(g.dtype)
            for g, s, p in zip(leaves, states, powers, strict=True)
        ]
        return unflatten_linear_layers(updates, treedef.unflatten(directions)), EmaGramState(states)

    return optax.GradientTransformation(init_fn, update_fn)


@OptimizerConfig.register_subclass("ema_gram_polar")
@dataclass(frozen=True)
class EmaGramConfig(OptimizerConfig):
    """EMA-Gram matrices and fixed AdamW nonmatrices, with one global clip first."""

    alpha: float = 0.25
    ridge: float = 1e-4
    momentum: float = 0.95
    gram_beta: float = 0.99
    polar_steps: int = 5
    eigh_layout: str = "replicated"
    """Matrix shards eigendecomposition; matrix_power also reconstructs each power on that shard."""
    eigh_implementation: str = "default"
    """Backend default or full Jacobi; selection never changes Gram geometry."""
    adam_lr: float = 3e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.9999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 0.1

    def build(self, num_train_steps):
        def optimizer(learning_rate, adam_lr):
            matrices = optax.chain(
                scale_by_ema_gram(
                    alpha=self.alpha,
                    ridge=self.ridge,
                    beta1=self.momentum,
                    beta2=self.gram_beta,
                    polar_steps=self.polar_steps,
                    eigh_layout=self.eigh_layout,
                    eigh_implementation=self.eigh_implementation,
                ),
                optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()),
                optax.scale(-learning_rate),
            )
            nonmatrices = optax.adamw(
                adam_lr,
                b1=self.adam_beta1,
                b2=self.adam_beta2,
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay,
                mask=self.build_weight_decay_mask(),
            )
            # Reuse Muon's embedding/head/bias routing, without its aspect-ratio scaling.
            mask = partial(MuonConfig.create_mask, self, use_kimi_scaling=True)
            return optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.multi_transform({"muon": matrices, "adamw": nonmatrices}, mask),
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=self.lr_scheduler(num_train_steps),
            adam_lr=self.lr_scheduler(num_train_steps, override_lr=self.adam_lr),
        )
