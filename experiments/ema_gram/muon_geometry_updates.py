# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete Muon probe updates at the approved reference learning rate."""

from dataclasses import replace

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import optax
from levanter.optim.util import flatten_linear_layers, unflatten_linear_layers

from experiments.ema_gram.muon_geometry_kernels import f5_finite_difference, matrix_candidates
from experiments.speedrun.prism_berkeley_qwen3_scaling.muon_error_feedback_optimizer import (
    NuclearHessianSettings,
    ScaleByErrorAwareMuonState,
)


def momentum_state(state):
    matches = [
        x
        for x in jax.tree.leaves(state, is_leaf=lambda v: isinstance(v, ScaleByErrorAwareMuonState))
        if isinstance(x, ScaleByErrorAwareMuonState)
    ]
    if len(matches) != 1:
        raise ValueError("Require exactly one complete pinned Muon momentum state")
    return matches[0]


def native_muon_functions(cfg, reference_lr):
    original = cfg.optimizer
    if original.momentum != 0.98 or original.nesterov or original.use_kimi_scaling:
        raise ValueError("Pinned300M protocol expects normalized momentum0.98 and original Muon scaling")
    ratio = original.adam_lr / original.learning_rate
    reference = replace(
        original,
        learning_rate=reference_lr,
        adam_lr=reference_lr * ratio,
        lr_schedule="constant",
        min_lr_ratio=1.0,
        warmup=0,
        decay=None,
        rewarmup=0,
        cooldown=None,
    )

    def native_loss(model, batch):
        return (
            cfg.trainer.mp.cast_to_compute(model)
            .compute_next_token_loss(batch, key=None, logsumexp_weight=cfg.z_loss_weight)
            .scalar()
        )

    functions = {}
    for name, policy, gain in (
        ("muon", "muon", 0),
        ("blend_0.02", "blend", 0.02),
        ("blend_0.05", "blend", 0.05),
        ("hesscorr_0.05", "hesscorr", 0.05),
        ("hesscorr_0.3", "hesscorr", 0.3),
    ):
        config = replace(
            reference,
            policy=policy,
            blend_gain=gain if policy == "blend" else 0,
            correction_gain=gain if policy == "hesscorr" else 0,
        )
        functions[name] = hax.named_jit(
            config.build(cfg.trainer.num_train_steps).update, axis_resources=cfg.trainer.compute_axis_mapping
        )
    return (
        hax.named_jit(eqx.filter_value_and_grad(native_loss), axis_resources=cfg.trainer.compute_axis_mapping),
        functions,
        reference,
    )


def tangent_functions(cfg):
    settings = NuclearHessianSettings(
        cubic_steps=cfg.optimizer.cubic_steps,
        sylvester_steps=cfg.optimizer.sylvester_steps,
        inverse_steps=cfg.optimizer.inverse_steps,
        eps=cfg.optimizer.muon_epsilon,
    )

    def matrix_inputs(post_state, gradient):
        masked = momentum_state(post_state).momentum_buffer
        filtered = jax.tree.map(
            lambda m, g: optax.MaskedNode() if isinstance(m, optax.MaskedNode) else g,
            masked,
            gradient,
            is_leaf=lambda x: isinstance(x, optax.MaskedNode),
        )
        momenta, definition = jax.tree.flatten(flatten_linear_layers(masked))
        gradients = jax.tree.leaves(flatten_linear_layers(filtered))
        return masked, momenta, gradients, definition

    def tangent(post_state, gradient):
        masked, momenta, gradients, definition = matrix_inputs(post_state, gradient)
        output, records = [], []
        for m, g in zip(momenta, gradients, strict=True):
            shape = m.shape
            _, derivative, metrics = jax.vmap(lambda a, b: matrix_candidates(a, b, settings))(
                m.reshape((-1, *shape[-2:])), g.reshape((-1, *shape[-2:]))
            )
            scale = jnp.sqrt(jnp.maximum(1.0, shape[-2] / shape[-1]))
            output.append(derivative.reshape(shape) * scale)
            records.append(metrics)
        return unflatten_linear_layers(masked, definition.unflatten(output)), tuple(records)

    def difference(post_state, gradient, epsilon):
        masked, momenta, gradients, definition = matrix_inputs(post_state, gradient)
        output = []
        for m, g in zip(momenta, gradients, strict=True):
            shape = m.shape
            derivative, _ = jax.vmap(lambda a, b: f5_finite_difference(a, b, epsilon))(
                m.reshape((-1, *shape[-2:])), g.reshape((-1, *shape[-2:]))
            )
            output.append(derivative.reshape(shape) * jnp.sqrt(jnp.maximum(1.0, shape[-2] / shape[-1])))
        return unflatten_linear_layers(masked, definition.unflatten(output))

    mapping = cfg.trainer.compute_axis_mapping
    return hax.named_jit(tangent, axis_resources=mapping), hax.named_jit(difference, axis_resources=mapping)


def add_f5_feedback(plain_update, tangent, reference_lr):
    return jax.tree.map(
        lambda t, u: u if isinstance(t, optax.MaskedNode) else u - 0.05 * reference_lr * t,
        tangent,
        plain_update,
        is_leaf=lambda x: isinstance(x, optax.MaskedNode),
    )
