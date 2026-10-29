# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Path B native training/export workers; no submission, retries, or capacity checks."""

import hashlib
import io
import json
from collections.abc import Callable
from dataclasses import asdict, replace

import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pyarrow.parquet as pq
from levanter.compat.hf_export import run_on_export_writer, write_export_json
from levanter.compute_dtype import cast_model_for_compute
from levanter.data.text.datasets import LmDataConfig
from levanter.main.train_lm import preserve_hf_runtime_settings
from levanter.models.snowball import GrugMoeHfConfig, snowball_loss_with_dispatch_stats
from levanter.tracker import NoopConfig
from levanter.trainer import Trainer, initialize
from levanter.trainer_state import FrozenParameterStorage, cast_params_by_trainability
from rigging.filesystem.storage_path import StoragePath

from experiments.interp.snowball_identity.levanter_positions import router_dot_dtypes, router_dot_is_fp32
from experiments.post_training.identity_context_distill.lr0_probe import check_pod_storage
from experiments.post_training.identity_context_distill.precision_source_audit import check_export_auxiliary_metadata
from experiments.post_training.identity_context_distill.runtime_recipe import (
    check_precision_runtime_receipt,
    check_xla_precision,
    configure_precision_runtime,
    xla_precision_in_process,
)
from experiments.post_training.identity_context_distill.sft import (
    BASE_MODEL,
    check_export_metadata,
    source_export_overrides,
)
from experiments.post_training.identity_context_distill.sft_launch import sft_runtime_receipt, source_bind
from experiments.post_training.identity_path_b.checkpoints import AsyncCheckpoints, CheckpointEvent
from experiments.post_training.identity_path_b.native import packed_dataset
from experiments.post_training.identity_path_b.recipe import freeze_special_rows, path_b_config, special_row_values
from experiments.post_training.identity_path_b.rehydrate import restore_trained_model
from experiments.post_training.identity_path_b.smoke import check_training_exposure
from experiments.post_training.identity_path_b.storage_audit import (
    assert_special_row_bits,
    audit_export_storage,
    audit_initialized_storage,
)
from experiments.post_training.identity_path_b.train import run_updates


def read_pinned(path: str, expected_sha256: str) -> bytes:
    raw = StoragePath(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Worker input hash differs: {path}")
    return raw


def runtime_config(*, run_root: str, run_id: str, updates: int, runtime_bytes: bytes, runtime_sha256: str):
    """Retain CD16's approved numerical recipe; packed data is separately hash-bound."""
    receipt = sft_runtime_receipt(runtime_bytes, runtime_sha256)
    check_xla_precision(receipt["xla_precision"])
    check_pod_storage()
    source_bytes = (StoragePath(BASE_MODEL) / "config.json").read_bytes()
    generation_bytes = (StoragePath(BASE_MODEL) / "generation_config.json").read_bytes()
    source, generation = json.loads(source_bytes), json.loads(generation_bytes)
    # This config supplies native tokenizer metadata only. The actual train set
    # is packed_dataset below, never the ChatDataset reader or factory placeholder.
    tokenizer_config = LmDataConfig(tokenizer=BASE_MODEL, components={})
    config = path_b_config(
        data=tokenizer_config,
        updates=updates,
        output_uri=run_root,
        run_id=run_id,
        source_config=source,
        source_generation_config=generation,
    )
    reference = receipt["routing_reference"]
    rounding = read_pinned(reference["path"], reference["sha256"])
    config, observed = configure_precision_runtime(
        config,
        config_bytes=source_bytes,
        generation_bytes=generation_bytes,
        rounding_bytes=rounding,
        rounding_path=reference["path"],
        rounding_sha256=reference["sha256"],
        xla_precision=xla_precision_in_process(),
    )
    if observed != receipt:
        raise ValueError("Path B runtime model/precision differs from the approved CD receipt")
    config = replace(config, trainer=replace(config.trainer, tracker=NoopConfig(), load_checkpoint=False))
    check_precision_runtime_receipt(
        config, receipt, config_bytes=source_bytes, generation_bytes=generation_bytes, rounding_bytes=rounding
    )
    return config, receipt, source, generation


def load_source(config, receipt: dict, source: dict, trainer):
    converter = config.model.hf_checkpoint_converter().replaced(
        reference_checkpoint=BASE_MODEL,
        tokenizer=config.data.the_tokenizer,
    )
    reloaded = converter.config_from_hf_config(GrugMoeHfConfig(**source))
    reloaded = preserve_hf_runtime_settings(reloaded, config.model, config.trainer.compute_dtype_overrides)
    model = converter.load_pretrained(
        config.model.model_type, config=reloaded, axis_mapping=trainer.parameter_axis_mapping, dtype=None
    )
    if asdict(model.config) != receipt["runtime_model_config"]:
        raise ValueError("HF reload changed the reviewed runtime model config")
    return model, converter


def admit_source(config, model, *, source_receipt: dict, source_objects: dict, run_root: str):
    raw = read_pinned(source_receipt["path"], source_receipt["sha256"])
    objects = json.loads(read_pinned(source_objects["path"], source_objects["sha256"]))
    run_on_export_writer(
        lambda: write_export_json(
            StoragePath(run_root) / "audits/source-bind.json",
            source_bind(json.loads(raw), objects),
        )
    )
    audit_initialized_storage(model, raw, source_receipt["sha256"], f"{run_root}/audits/source-storage.json")
    if config.trainer.frozen_parameter_storage != FrozenParameterStorage.NATIVE:
        raise ValueError("Path B fully frozen fields require native storage")
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
    checkpoint_hook_factory: Callable[[], AsyncCheckpoints],
) -> None:
    """All training ranks: admit source/actual packed data, update, commit, emit async events."""
    packed_raw = read_pinned(packed_path, packed_sha256)
    exposure = json.loads(read_pinned(exposure_path, exposure_sha256))
    check_training_exposure(exposure, arm)
    config, receipt, source, _ = runtime_config(
        run_root=run_root,
        run_id=run_id,
        updates=exposure["updates"],
        runtime_bytes=runtime_bytes,
        runtime_sha256=runtime_sha256,
    )
    initialize(config)
    if jax.process_count() != 4 or jax.device_count() != 32:
        raise ValueError("Path B scientific mesh requires4 processes/32 devices")
    optimizer = freeze_special_rows(config.optimizer.build(config.trainer.num_train_steps))
    with Trainer(config.trainer, optimizer, snowball_loss_with_dispatch_stats, add_default_hooks=False) as trainer:
        model, _ = load_source(config, receipt, source, trainer)
        state = trainer.initial_state(
            jrandom.PRNGKey(310), model=model, is_trainable=config.adapter.trainable_filter(model)
        )
        del model
        admit_source(
            config, state.model, source_receipt=source_receipt, source_objects=source_objects, run_root=run_root
        )
        with hax.partitioning.set_mesh(trainer.device_mesh), hax.axis_mapping(trainer.compute_axis_mapping):
            view = cast_model_for_compute(state.model, trainer.mp, config.trainer.compute_dtype_overrides)
            dot = router_dot_dtypes(view, trainer.mp.compute_dtype)
        del view
        if not router_dot_is_fp32(dot):
            raise ValueError("Path B router-dot witness is not FP32")
        rows = pq.read_table(io.BytesIO(packed_raw)).to_pylist()
        pos = config.model.max_Pos.resize(config.train_seq_len)
        dataset = packed_dataset(packed_raw, packed_sha256, pos)
        batches = trainer.data_loader(dataset).iter_from_step(0)
        state, events = run_updates(
            trainer=trainer,
            state=state,
            batches=batches,
            packed_rows=rows,
            exposure=exposure,
            arm=arm,
            run_root=run_root,
            training_pin=training_pin,
            packed_data_sha256=packed_sha256,
            checkpoint_hook=checkpoint_hook_factory(),
        )
        result = {
            "completed_updates": int(state.step),
            "arm": arm,
            "training_pin": training_pin,
            "packed_sha256": packed_sha256,
            "exposure_sha256": exposure_sha256,
            "source_receipt": source_receipt,
            "runtime_receipt_sha256": runtime_sha256,
            "router_dot_dtypes": dot,
            "checkpoints": events,
            "exports_and_evaluations": "independent jobs; results may remain pending",
        }
        run_on_export_writer(lambda: write_export_json(StoragePath(run_root) / "training-result.json", result))


def export_worker(
    *,
    event: CheckpointEvent,
    run_id: str,
    runtime_bytes: bytes,
    runtime_sha256: str,
    source_receipt: dict,
    source_objects: dict,
) -> None:
    """Independent ranks: source rehydration + strict trainable overlay + complete HF audit."""
    event.payload()  # Includes the owner-mandated temporary checkpoint/HF/eval roots.
    run_root = event.hf_export.rsplit("/hf/", 1)[0]
    if (StoragePath(event.hf_export) / "config.json").exists():
        raise FileExistsError("Independent export destination already contains a model")
    config, receipt, source, generation = runtime_config(
        run_root=run_root,
        run_id=run_id,
        updates=event.completed_updates,
        runtime_bytes=runtime_bytes,
        runtime_sha256=runtime_sha256,
    )
    initialize(config)
    if jax.process_count() != 4 or jax.device_count() != 32:
        raise ValueError("Independent export uses the reviewed4-process/32-device source mesh")
    with Trainer(
        config.trainer,
        config.optimizer.build(config.trainer.num_train_steps),
        snowball_loss_with_dispatch_stats,
        add_default_hooks=False,
    ) as trainer:
        model, converter = load_source(config, receipt, source, trainer)
        mask = config.adapter.trainable_filter(model)
        model = cast_params_by_trainability(
            model, trainer.mp, mask, frozen_parameter_storage=FrozenParameterStorage.NATIVE
        )
        source_raw = admit_source(
            config,
            model,
            source_receipt=source_receipt,
            source_objects=source_objects,
            run_root=f"{event.hf_export}/admission",
        )
        rows = special_row_values(model)
        trained = restore_trained_model(
            model,
            mask,
            event.native_checkpoint,
            completed_updates=event.completed_updates,
            axis_mapping=trainer.parameter_axis_mapping,
            mesh=trainer.device_mesh,
        )
        assert_special_row_bits(trained, rows)
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
                source,
                json.loads((StoragePath(event.hf_export) / "config.json").read_bytes()),
            )
        )
        run_on_export_writer(lambda: check_export_auxiliary_metadata(BASE_MODEL, event.hf_export))
        audit_export_storage(trained, event.hf_export, source_raw, source_receipt["sha256"])
        run_on_export_writer(
            lambda: write_export_json(
                StoragePath(event.hf_export) / "path-b-checkpoint-event.json",
                event.payload(),
            )
        )
