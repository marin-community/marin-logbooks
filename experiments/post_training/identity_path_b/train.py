# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finite native Path B updates, bit-preserving row restore, and checkpoint events."""

import hashlib
from collections.abc import Iterator
from functools import partial

import jax
import numpy as np
from levanter.compat.hf_export import run_on_export_writer
from levanter.tracker import current_tracker

from experiments.interp.snowball_identity.multihost import gather_global
from experiments.post_training.identity_context_distill.sft_launch import DispatchMetrics
from experiments.post_training.identity_path_b.checkpoints import AsyncCheckpoints, CheckpointEvent
from experiments.post_training.identity_path_b.recipe import (
    FROZEN_FIELDS,
    restore_frozen_state,
    special_row_values,
)


def frozen_field_bits(model) -> dict:
    """Hash local shards of every fully frozen parameter without numeric coercion."""
    result = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(model)[0]:
        fields = {key.name for key in path if isinstance(key, jax.tree_util.GetAttrKey)}
        if not fields.intersection(FROZEN_FIELDS):
            continue
        name = jax.tree_util.keystr(path)
        result[name] = [
            {
                "index": str(shard.index),
                "dtype": str(shard.data.dtype),
                "sha256": hashlib.sha256(np.asarray(shard.data).tobytes()).hexdigest(),
            }
            for shard in leaf.addressable_shards
        ]
    if len(result) != 104:
        raise ValueError(f"Expected Q/K/router/bias leaves in all26 layers, found {len(result)}")
    return result


def check_consumed_batch(batch, expected: list[dict]) -> None:
    """Compare the actual global arrays to reviewed tokens/weights/query/key segments."""
    query, keys = batch.attn_mask.segment_ids
    tokens, weights, qseg, kseg = (
        gather_global(array) for array in [batch.tokens.array, batch.loss_weight.array, query.array, keys.array]
    )
    reference_tokens = np.asarray([r["input_ids"] for r in expected], dtype=np.int32)
    reference_segments = np.asarray([r["segment_ids"] for r in expected], dtype=np.int32)
    reference_weights = np.roll(np.asarray([r["target_weights"] for r in expected], dtype=np.float32), -1, axis=-1)
    reference_weights[:, -1] = 0
    reference_weights *= np.roll(reference_segments, -1, axis=-1) >= 0
    np.testing.assert_array_equal(tokens, reference_tokens)
    np.testing.assert_array_equal(weights, reference_weights)
    np.testing.assert_array_equal(qseg, reference_segments)
    np.testing.assert_array_equal(kseg, reference_segments)


def run_updates(
    *,
    trainer,
    state,
    batches: Iterator,
    packed_rows: list[dict],
    exposure: dict,
    arm: str,
    run_root: str,
    training_pin: str,
    packed_data_sha256: str,
    checkpoint_hook: AsyncCheckpoints,
) -> tuple:
    """Run only after source/master/precision admission and distributed initialization.

    Native checkpoints commit on training ranks. Export/evaluation is submitted
    independently from immutable checkpoint events; their results never block
    this loop. The caller must use the row-masked optimizer and no default save hooks.
    """
    updates = exposure["updates"]
    if trainer.config.train_batch_size != 8 or trainer.config.microbatch_size != 8:
        raise ValueError("Weighted loss requires one logical global batch, with no microbatch ratio averaging")
    if int(state.step) != 0 or len(packed_rows) != updates * 8:
        raise ValueError("Expected fresh state and the exact finite packed arm")
    rows = special_row_values(state.model)
    frozen = frozen_field_bits(state.model)
    checkpointer = trainer.config.checkpointer.create(trainer.run_id)
    checkpoints = set(exposure["checkpoint_updates"])
    records = []
    for update in range(updates):
        batch = next(batches)
        check_consumed_batch(batch, packed_rows[update * 8 : (update + 1) * 8])
        dispatch = DispatchMetrics()
        with current_tracker(dispatch):
            info = trainer.train_step(state, batch)
        counts = dispatch.checked_counts()
        # Zero updates alone change -0.0 to+0.0 under optax.apply_updates.
        # Restore the original VALUES before another forward or any checkpoint.
        state = restore_frozen_state(info.state, rows)
        completed = int(state.step)
        if completed != update + 1 or not np.isfinite(info.loss):
            raise ValueError("Native update did not advance with a finite loss")
        if completed not in checkpoints:
            continue
        if frozen_field_bits(state.model) != frozen:
            raise ValueError("A frozen Q/K/router/bias parameter changed bits")
        label = completed - 1
        checkpointer.on_step(tree=state.saveable_state, step=label, force=True)
        checkpointer.wait_until_finished()
        native = f"{trainer.config.checkpointer.expanded_path(trainer.run_id)}/step-{label}"
        event = CheckpointEvent(
            completed,
            native,
            f"{run_root}/hf/updates-{completed}",
            exposure["cumulative_batches"][completed - 1],
            training_pin,
            packed_data_sha256,
            arm,
        )
        run_on_export_writer(partial(checkpoint_hook.publish_committed, event))
        records.append(
            {
                "completed_updates": completed,
                "loss": float(info.loss),
                "dispatch": counts,
                "native_checkpoint": native,
                "exposure": event.exposure,
            }
        )
    run_on_export_writer(checkpoint_hook.finish_submissions)
    return state, records


def trend_row(event: dict, scores: dict) -> str:
    """Format one explicitly incomplete or complete diagnostic checkpoint row."""
    fields = [event["arm"], str(event["completed_updates"])]
    fields += [str(scores.get(k, "pending")) for k in ["name", "creator", "foreign", "third_person", "gsm8k", "bfcl"]]
    return "| " + " | ".join(fields) + " |"
