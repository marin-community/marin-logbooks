# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Optimizer comparison on the same stacked dense Grug parameter groups."""

from dataclasses import dataclass
from enum import StrEnum

import jax
import optax
from levanter.optim.config import OptimizerConfig
from levanter.utils.jax_utils import leaf_key_paths

from experiments.grug.fast_track.grugmuon_stacked import _grug_scale_with_muon
from experiments.grug.fast_track.optimizer import (
    GrugMoeMuonHConfig,
    _match_named_update_sharding,
    _scale_invariant_hyperball_updates,
    scale_with_grug_muonh,
)


class DenseOptimizer(StrEnum):
    ADAMW = "adamw"
    MUON = "muon"
    ADAMH = "adamh"
    MUONH = "muonh"


def _scale_by_dense_adamh(beta1: float, beta2: float, epsilon: float, learning_rate: float):
    adam = optax.scale_by_adam(beta1, beta2, epsilon)

    def update(updates, state, params):
        directions, state = adam.update(updates, state, params)
        return _scale_invariant_hyperball_updates(params, directions, learning_rate), state

    return optax.GradientTransformation(adam.init, update)


@OptimizerConfig.register_subclass("grug_dense_comparison")
@dataclass(frozen=True)
class DenseOptimizerConfig(OptimizerConfig):
    """Four optimizers with shared dense matrix, head, and Adam fallback groups.

    Attention, MLP, and GatedNorm matrices use the selected optimizer. The head uses
    AdamH for the H variants and AdamW otherwise. Embeddings, norm gains, attention
    gates, and SConv kernels use Adam. The W variants decay matrices, head, and
    embeddings; the H variants preserve matrix norms and apply no weight decay.
    """

    optimizer: DenseOptimizer = DenseOptimizer.ADAMW
    adam_lr: float = 6e-4
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 5
    muon_epsilon: float = 1e-8
    max_grad_norm: float | None = None

    def __post_init__(self):
        if self.optimizer in (DenseOptimizer.ADAMH, DenseOptimizer.MUONH) and self.weight_decay != 0:
            raise ValueError("AdamH and MuonH require weight_decay=0; their matrix norms are preserved")

    def build(self, num_train_steps):
        matrix_schedule = self.lr_scheduler(num_train_steps)
        adam_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            adam = optax.scale_by_adam(self.beta1, self.beta2, self.epsilon)
            match self.optimizer:
                case DenseOptimizer.ADAMW:
                    matrix = optax.chain(adam, optax.add_decayed_weights(self.weight_decay), optax.scale(-learning_rate))
                case DenseOptimizer.MUON:
                    matrix = optax.chain(
                        _grug_scale_with_muon(
                            momentum=self.momentum,
                            nesterov=self.nesterov,
                            steps=self.backend_steps,
                            muon_eps=self.muon_epsilon,
                        ),
                        optax.add_decayed_weights(self.weight_decay),
                        optax.scale(-learning_rate),
                    )
                case DenseOptimizer.ADAMH:
                    matrix = _scale_by_dense_adamh(self.beta1, self.beta2, self.epsilon, learning_rate)
                case DenseOptimizer.MUONH:
                    matrix = scale_with_grug_muonh(
                        momentum=self.momentum,
                        nesterov=self.nesterov,
                        steps=self.backend_steps,
                        muon_eps=self.muon_epsilon,
                        learning_rate=learning_rate,
                    )
                case _:
                    raise ValueError(f"Unknown dense optimizer: {self.optimizer}")

            if self.optimizer in (DenseOptimizer.ADAMH, DenseOptimizer.MUONH):
                head = _scale_by_dense_adamh(self.beta1, self.beta2, self.epsilon, learning_rate)
            else:
                head = optax.chain(adam, optax.add_decayed_weights(self.weight_decay), optax.scale(-adam_lr))
            fallback = optax.chain(adam, optax.scale(-adam_lr))
            embedding = optax.chain(adam, optax.add_decayed_weights(self.weight_decay), optax.scale(-adam_lr))
            return optax.multi_transform(
                {"matrix": matrix, "head": head, "adam": fallback, "embedding": embedding}, self.create_mask
            )

        transform = optax.inject_hyperparams(optimizer)(learning_rate=matrix_schedule, adam_lr=adam_schedule)
        if self.max_grad_norm is not None:
            transform = optax.chain(optax.clip_by_global_norm(self.max_grad_norm), transform)
        return optax.chain(transform, _match_named_update_sharding())

    def create_mask(self, params):
        groups = GrugMoeMuonHConfig().create_mask(params)
        paths = leaf_key_paths(params)

        def label(group, path):
            if "token_embed" in path:
                return "embedding"
            return {"muonh": "matrix", "adamh": "head", "adam": "adam"}[group]

        return jax.tree.map(label, groups, paths)


def build_dense_optimizer_config(
    *,
    optimizer: DenseOptimizer,
    learning_rate: float,
    adam_lr: float,
    weight_decay: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
    epsilon: float = 1e-8,
    warmup: float = 0.01,
    lr_schedule: str = "cosine",
    min_lr_ratio: float = 0.1,
) -> DenseOptimizerConfig:
    """Build an explicitly parameterized optimizer for a dense comparison run."""
    return DenseOptimizerConfig(
        optimizer=optimizer,
        learning_rate=learning_rate,
        adam_lr=adam_lr,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        warmup=warmup,
        lr_schedule=lr_schedule,
        min_lr_ratio=min_lr_ratio,
    )
