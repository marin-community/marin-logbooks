# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit migration of the July momentum-only Muon checkpoint schema."""

from typing import NamedTuple

import equinox as eqx
import jax
import optax
from levanter.checkpoint import load_checkpoint

from experiments.speedrun.prism_berkeley_qwen3_scaling.muon_error_feedback_optimizer import ScaleByErrorAwareMuonState


class JulyMuonState(NamedTuple):
    """Exact state fields in the archived clean 7cd118640 implementation."""

    momentum_buffer: optax.Updates


def july_template(template):
    matches = [
        x
        for x in jax.tree.leaves(template.opt_state, is_leaf=lambda x: isinstance(x, ScaleByErrorAwareMuonState))
        if isinstance(x, ScaleByErrorAwareMuonState)
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one pinned Muon state")
    replacement = jax.tree.map(
        lambda x: JulyMuonState(x.momentum_buffer) if isinstance(x, ScaleByErrorAwareMuonState) else x,
        template.opt_state,
        is_leaf=lambda x: isinstance(x, ScaleByErrorAwareMuonState),
    )
    return eqx.tree_at(lambda x: x.opt_state, template, replacement)


def migrate_july_state(restored, expected_counter):
    """Use saved global/Adam/schedule counters; retain every saved parameter and moment."""
    counters = dict(
        step=int(restored.step),
        optimizer=int(restored.opt_state.count),
        adam=int(restored.opt_state.inner_state.inner_states["adamw"].inner_state[1].count),
    )
    counters.update(
        {"schedule_" + name: int(value.count) for name, value in restored.opt_state.hyperparams_states.items()}
    )
    if set(counters.values()) != {expected_counter}:
        raise ValueError(f"Cannot migrate inconsistent saved counters: {counters}")
    matches = [
        x
        for x in jax.tree.leaves(restored.opt_state, is_leaf=lambda x: isinstance(x, JulyMuonState))
        if isinstance(x, JulyMuonState)
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one legacy momentum-only state")
    replacement = jax.tree.map(
        lambda x: (
            ScaleByErrorAwareMuonState(count=restored.opt_state.count, momentum_buffer=x.momentum_buffer)
            if isinstance(x, JulyMuonState)
            else x
        ),
        restored.opt_state,
        is_leaf=lambda x: isinstance(x, JulyMuonState),
    )
    migrated = eqx.tree_at(lambda x: x.opt_state, restored, replacement)
    return migrated, dict(
        schema="momentum_only_7cd118640",
        saved_counters=counters,
        added_field="Muon correction-warmup counter copied from agreeing saved optimizer counter",
        scope="Explicit schema migration; all saved arrays restored strictly; no zero initialization",
    )


def restore_muon_checkpoint(template, spec, axis_mapping):
    schema = spec["checkpoint_schema"]
    if schema == "counter_fffd6a64":
        return load_checkpoint(template, spec["checkpoint"], axis_mapping=axis_mapping, allow_partial=False), dict(
            schema=schema
        )
    if schema != "momentum_only_7cd118640":
        raise ValueError("Unknown Muon checkpoint schema")
    restored = load_checkpoint(
        july_template(template), spec["checkpoint"], axis_mapping=axis_mapping, allow_partial=False
    )
    return migrate_july_state(restored, spec["expected_counter"])
