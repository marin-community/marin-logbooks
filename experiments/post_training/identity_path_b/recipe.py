# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Path B optimizer constraints and native recipe; no launch side effects."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import optax
from levanter.adaptor import FrozenParameterAdaptorConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.trainer_state import TrainerState

from experiments.post_training.identity_context_distill.sft import identity_sft_config
from experiments.post_training.identity_path_b.data import END_IDS, LOW_WEIGHT_IDS

FROZEN_FIELDS = ("w_q", "w_k", "router", "router_bias")
FROZEN_TOKEN_IDS = tuple(sorted(LOW_WEIGHT_IDS | END_IDS))


def mask_special_rows(tree):
    """Zero selected vocabulary rows, respecting the native head's transposition."""
    ids = jnp.asarray(FROZEN_TOKEN_IDS)

    def mask(path, leaf):
        if leaf is None or not path or not isinstance(path[-1], jax.tree_util.GetAttrKey):
            return leaf
        field = path[-1].name
        if field == "token_embed":
            return leaf.at[ids, :].set(0)
        if field == "output_proj":
            return leaf.at[:, ids].set(0)
        return leaf

    return jax.tree_util.tree_map_with_path(mask, tree)


def freeze_special_rows(optimizer: optax.GradientTransformation) -> optax.GradientTransformation:
    """Mask gradients AND post-optimizer updates, including momentum/weight decay.

    Full fields are excluded by the native adaptor. Partial rows remain in its
    parameter tree, so a gradient mask alone would not freeze AdamW updates.
    """

    def init(params):
        return optimizer.init(params)

    def update(grads, state, params=None, **extra_args):
        updates, state = optimizer.update(mask_special_rows(grads), state, params, **extra_args)
        return mask_special_rows(updates), state

    return optax.GradientTransformationExtraArgs(init, update)


def special_row_values(model) -> dict[str, jax.Array]:
    """Snapshot small row VALUES before donating the initial full parameter tree."""
    ids = jnp.asarray(FROZEN_TOKEN_IDS)
    values = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(model)[0]:
        if not path or not isinstance(path[-1], jax.tree_util.GetAttrKey):
            continue
        field = path[-1].name
        if field in ("token_embed", "output_proj"):
            axis = 0 if field == "token_embed" else 1
            values[field] = jnp.take(leaf, ids, axis=axis).copy()
    if set(values) != {"token_embed", "output_proj"}:
        raise ValueError("Expected exactly one native embedding and output projection")
    for value in values.values():
        value.block_until_ready()
    return values


def restore_special_row_values(model, values: dict[str, jax.Array]):
    """Restore original bits, including signed zero, after optimizer application."""
    ids = jnp.asarray(FROZEN_TOKEN_IDS)

    def restore(path, leaf):
        if not path or not isinstance(path[-1], jax.tree_util.GetAttrKey):
            return leaf
        field = path[-1].name
        if field == "token_embed":
            return leaf.at[ids, :].set(values[field])
        if field == "output_proj":
            return leaf.at[:, ids].set(values[field])
        return leaf

    return jax.tree_util.tree_map_with_path(restore, model)


def restore_frozen_state(state: TrainerState, values: dict[str, jax.Array]) -> TrainerState:
    """Required immediately after every native train step, before use or saving."""
    return replace(state, model=restore_special_row_values(state.model, values))


def loss_token_weights(tokens, binary_target_mask):
    """Return target-position weights; native ChatDataset shifts exactly once."""
    low = jnp.isin(tokens, jnp.asarray(tuple(LOW_WEIGHT_IDS)))
    return binary_target_mask.astype(jnp.float32) * jnp.where(low, 0.1, 1.0)


def path_b_config(
    *, data, updates: int, output_uri: str, run_id: str, source_config: dict, source_generation_config: dict
) -> TrainLmConfig:
    """Keep the reviewed CD16 precision/mesh recipe, changing the authorized knobs."""
    base = identity_sft_config(
        dataset_uri="path-b-reviewed-input",
        cache_uri="path-b-reviewed-cache",
        output_uri=output_uri,
        run_id=run_id,
        seed=310,
        hosts=4,
        source_config=source_config,
        source_generation_config=source_generation_config,
        moe_capacity_factor=4.0,
        router_combine_fp32=True,
        fp32_activations=True,
        attention_implementation="torch_order",
        stop_after_updates=16,
    )
    adapter = FrozenParameterAdaptorConfig(
        frozen_field_names=FROZEN_FIELDS,
        hf_save_dtype_overrides=base.adapter.hf_save_dtype_overrides,
        hf_config_overrides=base.adapter.hf_config_overrides,
    )
    return replace(
        base,
        data=data,
        adapter=adapter,
        trainer=replace(base.trainer, num_train_steps=updates),
        hf_save_steps=None,
    )
