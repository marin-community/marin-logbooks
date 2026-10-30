# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Final-only BFCL ablation workers; no live calls outside native entrypoints."""

import hashlib
import io
import json
from dataclasses import replace
from enum import StrEnum
from functools import partial

import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pyarrow.parquet as pq
from levanter.compat.hf_export import run_on_export_writer, write_export_json
from levanter.compute_dtype import cast_model_for_compute
from levanter.models.snowball import snowball_loss_with_dispatch_stats
from levanter.tracker import current_tracker
from levanter.trainer import Trainer, initialize
from levanter.trainer_state import FrozenParameterStorage, cast_params_by_trainability
from rigging.filesystem.storage_path import StoragePath

from experiments.interp.snowball_identity.levanter_positions import router_dot_dtypes, router_dot_is_fp32
from experiments.post_training.identity_context_distill.precision_source_audit import check_export_auxiliary_metadata
from experiments.post_training.identity_context_distill.sft import (
    BASE_MODEL,
    check_export_metadata,
    source_export_overrides,
)
from experiments.post_training.identity_context_distill.sft_launch import DispatchMetrics, source_bind
from experiments.post_training.identity_path_b.ablation_data import UPDATES
from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent, event_bytes
from experiments.post_training.identity_path_b.native import packed_dataset
from experiments.post_training.identity_path_b.recipe import (
    FROZEN_FIELDS,
    freeze_special_rows,
    restore_frozen_state,
    special_row_values,
)
from experiments.post_training.identity_path_b.rehydrate import restore_trained_model, source_restore_template
from experiments.post_training.identity_path_b.retention import save_temporary_checkpoint
from experiments.post_training.identity_path_b.storage_audit import (
    FROZEN_HF_NAMES,
    assert_special_row_bits,
    audit_export_storage,
    audit_initialized_storage,
)
from experiments.post_training.identity_path_b.train import check_consumed_batch
from experiments.post_training.identity_path_b.worker import export_mesh, load_source, read_pinned, runtime_config


class AblationArm(StrEnum):
    CONTROL = "A0"
    TRAIN_SPECIAL_ROWS = "A1"
    REPLAY_ONLY = "A2"
    IDENTITY_OFF = "A3"
    T3_BRIDGE = "A4"
    GLM_ON_OFF_FALLBACK = "A6"
    GLM_ON_TENTH = "A8a"
    GLM_ON_THREE_TENTHS = "A8b"
    GLM_ON_TENTH_72 = "A9"
    TULU_STYLE_44 = "A10"


def updates_for(arm: AblationArm) -> int:
    return 72 if arm == AblationArm.GLM_ON_TENTH_72 else UPDATES


def frozen_names(arm: AblationArm) -> frozenset[str]:
    return (
        frozenset(name for name in FROZEN_HF_NAMES if name.endswith(".router.bias"))
        if arm == AblationArm.T3_BRIDGE
        else FROZEN_HF_NAMES
    )


def frozen_fields(arm: AblationArm) -> tuple[str, ...]:
    return ("router_bias",) if arm == AblationArm.T3_BRIDGE else FROZEN_FIELDS


def freezes_special_rows(arm: AblationArm) -> bool:
    return arm not in (AblationArm.TRAIN_SPECIAL_ROWS, AblationArm.T3_BRIDGE)


def optimizer_for(config, arm: AblationArm):
    optimizer = config.optimizer.build(updates_for(arm))
    return freeze_special_rows(optimizer) if freezes_special_rows(arm) else optimizer


def config_for(config, arm: AblationArm):
    if (
        config.optimizer.learning_rate != 2e-6
        or config.optimizer.lr_schedule != "constant"
        or config.optimizer.warmup != 2
    ):
        raise ValueError("Ablations require the exact r5/T3 constant2e-6/warmup2 trajectory")
    if (
        config.trainer.num_train_steps != updates_for(arm)
        or config.trainer.frozen_parameter_storage != FrozenParameterStorage.NATIVE
    ):
        raise ValueError("Ablations require the arm-specific fresh native-storage horizon")
    return replace(config, adapter=replace(config.adapter, frozen_field_names=frozen_fields(arm)))


def check_report(report: dict, arm: AblationArm) -> None:
    if (
        report["arm"] != arm
        or report["updates"] != updates_for(arm)
        or report["checkpoint_updates"] != [updates_for(arm)]
        or report["special_token_loss_weight"] != 1.0
        or report["batch_size"] != 8
        or report["sequence_length"] != 8192
        or len(report["cumulative_batches"]) != updates_for(arm)
        or report["weighted_mass"] != report["raw_supervised_tokens"]
    ):
        raise ValueError("Ablation report differs from the admitted arm-specific unit-weight design")
    if arm == AblationArm.REPLAY_ONLY and report["raw_supervised_tokens"].get("identity", 0):
        raise ValueError("Replay-only ablation contains identity targets")
    if arm == AblationArm.IDENTITY_OFF and report["mode_final_text_tokens"].get("identity/on", 0):
        raise ValueError("OFF ablation contains identity ON occurrences")


def full_field_bits(model, arm: AblationArm) -> dict:
    values = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(model)[0]:
        fields = {key.name for key in path if isinstance(key, jax.tree_util.GetAttrKey)}
        if fields.intersection(frozen_fields(arm)):
            values[jax.tree_util.keystr(path)] = [
                (str(shard.index), str(shard.data.dtype), hashlib.sha256(np.asarray(shard.data).tobytes()).hexdigest())
                for shard in leaf.addressable_shards
            ]
    if len(values) != len(frozen_names(arm)):
        raise ValueError("Full-field freeze set differs from the arm's exact26/104 tensors")
    return values


def admit(model, source_receipt: dict, source_objects: dict, root: str, arm: AblationArm) -> bytes:
    raw = read_pinned(source_receipt["path"], source_receipt["sha256"])
    objects = json.loads(read_pinned(source_objects["path"], source_objects["sha256"]))
    run_on_export_writer(
        lambda: write_export_json(StoragePath(root) / "audits/source-bind.json", source_bind(json.loads(raw), objects))
    )
    audit_initialized_storage(
        model, raw, source_receipt["sha256"], f"{root}/audits/source-storage.json", frozen_names=frozen_names(arm)
    )
    return raw


def train_worker(
    *,
    run_root: str,
    run_id: str,
    arm: str,
    training_pin: str,
    packed_path: str,
    packed_sha256: str,
    exposure_path: str,
    exposure_sha256: str,
    runtime_bytes: bytes,
    runtime_sha256: str,
    source_receipt: dict,
    source_objects: dict,
) -> None:
    variant = AblationArm(arm)
    updates = updates_for(variant)
    packed = read_pinned(packed_path, packed_sha256)
    exposure = json.loads(read_pinned(exposure_path, exposure_sha256))
    check_report(exposure, variant)
    rows = pq.read_table(io.BytesIO(packed)).to_pylist()
    if len(rows) != updates * 8:
        raise ValueError("Ablation must consume all admitted packed examples at its fixed horizon")
    config, receipt, source, _ = runtime_config(
        run_root=run_root, run_id=run_id, updates=updates, runtime_bytes=runtime_bytes, runtime_sha256=runtime_sha256
    )
    config = config_for(config, variant)
    initialize(config)
    if jax.process_count() != 4 or jax.device_count() != 32:
        raise ValueError("Ablation training requires the unchanged4-host32-device mesh")
    with Trainer(
        config.trainer, optimizer_for(config, variant), snowball_loss_with_dispatch_stats, add_default_hooks=False
    ) as trainer:
        model, _ = load_source(config, receipt, source, trainer)
        state = trainer.initial_state(
            jrandom.PRNGKey(310), model=model, is_trainable=config.adapter.trainable_filter(model)
        )
        del model
        admit(state.model, source_receipt, source_objects, run_root, variant)
        frozen = full_field_bits(state.model, variant)
        special = special_row_values(state.model) if freezes_special_rows(variant) else None
        with hax.partitioning.set_mesh(trainer.device_mesh), hax.axis_mapping(trainer.compute_axis_mapping):
            view = cast_model_for_compute(state.model, trainer.mp, config.trainer.compute_dtype_overrides)
            dot = router_dot_dtypes(view, trainer.mp.compute_dtype)
        del view
        if not router_dot_is_fp32(dot):
            raise ValueError("Ablation router-dot compute is notFP32")
        dataset = packed_dataset(packed, packed_sha256, config.model.max_Pos.resize(config.train_seq_len))
        batches = trainer.data_loader(dataset).iter_from_step(0)
        records = []
        for update in range(updates):
            batch = next(batches)
            check_consumed_batch(batch, rows[update * 8 : (update + 1) * 8])
            dispatch = DispatchMetrics()
            with current_tracker(dispatch):
                info = trainer.train_step(state, batch)
            counts = dispatch.checked_counts()
            state = restore_frozen_state(info.state, special) if special is not None else info.state
            if int(state.step) != update + 1 or not np.isfinite(info.loss):
                raise ValueError("Ablation update failed its finite/progress check")
            records.append({"completed_updates": int(state.step), "loss": float(info.loss), "dispatch": counts})
        if full_field_bits(state.model, variant) != frozen:
            raise ValueError("Ablation changed a fully frozen parameter")
        special_hashes = assert_special_row_bits(state.model, special) if special is not None else None
        checkpointer = config.trainer.checkpointer.create(trainer.run_id)
        save_temporary_checkpoint(checkpointer, state.saveable_state, step=updates - 1)
        checkpointer.wait_until_finished()
        event = CheckpointEvent(
            completed_updates=updates,
            native_checkpoint=f"{run_root}/checkpoints/step-{updates - 1}",
            hf_export=f"{run_root}/hf/updates-{updates}",
            eval_scratch=f"{run_root}/eval/updates-{updates}",
            exposure=exposure["cumulative_batches"][-1],
            training_pin=training_pin,
            packed_data_sha256=packed_sha256,
            arm=arm,
        )
        result = {
            "arm": arm,
            "completed_updates": updates,
            "training_pin": training_pin,
            "packed_sha256": packed_sha256,
            "exposure_sha256": exposure_sha256,
            "special_rows_frozen": freezes_special_rows(variant),
            "frozen_special_row_sha256": special_hashes,
            "fully_frozen_hf_names": sorted(frozen_names(variant)),
            "updates": records,
        }
        run_on_export_writer(partial(write_export_json, StoragePath(run_root) / "training-result.json", result))
        run_on_export_writer(
            lambda: (StoragePath(run_root) / "final-checkpoint-event.json").write_bytes(event_bytes(event))
        )


def export_worker(
    *,
    event: CheckpointEvent,
    run_id: str,
    runtime_bytes: bytes,
    runtime_sha256: str,
    source_receipt: dict,
    source_objects: dict,
) -> None:
    variant = AblationArm(event.arm)
    updates = updates_for(variant)
    event.payload()
    if event.completed_updates != updates:
        raise ValueError("Ablations export only their exact arm-specific final native step")
    root = event.hf_export.rsplit("/hf/", 1)[0]
    if (StoragePath(event.hf_export) / "config.json").exists():
        raise FileExistsError("Ablation HF destination already holds a model")
    config, receipt, source, generation = runtime_config(
        run_root=root, run_id=run_id, updates=updates, runtime_bytes=runtime_bytes, runtime_sha256=runtime_sha256
    )
    config = config_for(config, variant)
    config = replace(config, trainer=replace(config.trainer, mesh=export_mesh()))
    initialize(config)
    if jax.process_count() != 1 or jax.device_count() != 8:
        raise ValueError("Ablation exports require one8-GPU node")
    with Trainer(
        config.trainer, config.optimizer.build(updates), snowball_loss_with_dispatch_stats, add_default_hooks=False
    ) as trainer:
        model, converter = load_source(config, receipt, source, trainer)
        mask = config.adapter.trainable_filter(model)
        model = cast_params_by_trainability(
            model, trainer.mp, mask, frozen_parameter_storage=FrozenParameterStorage.NATIVE
        )
        raw = admit(model, source_receipt, source_objects, f"{event.hf_export}/admission", variant)
        special = special_row_values(model) if freezes_special_rows(variant) else None
        template = source_restore_template(model, mask)
        del model
        trained = restore_trained_model(
            template,
            mask,
            event.native_checkpoint,
            completed_updates=updates,
            axis_mapping=trainer.parameter_axis_mapping,
            mesh=trainer.device_mesh,
        )
        del template
        if special is not None:
            assert_special_row_bits(trained, special)
        converter.with_config_overrides(source_export_overrides(source)).save_pretrained(
            trained,
            event.hf_export,
            dtype=jnp.bfloat16,
            dtype_overrides=config.adapter.hf_save_dtype_overrides,
            generation_config=generation,
            upload_to_hf=False,
        )
        run_on_export_writer(
            lambda: check_export_metadata(
                source, json.loads((StoragePath(event.hf_export) / "config.json").read_bytes())
            )
        )
        run_on_export_writer(lambda: check_export_auxiliary_metadata(BASE_MODEL, event.hf_export))
        audit_export_storage(trained, event.hf_export, raw, source_receipt["sha256"], frozen_names=frozen_names(variant))
        run_on_export_writer(
            partial(write_export_json, StoragePath(event.hf_export) / "path-b-checkpoint-event.json", event.payload())
        )
