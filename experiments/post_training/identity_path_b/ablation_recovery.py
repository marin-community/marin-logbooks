# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One authorized A1 export-only recovery, preserving its original training event."""

import json
import re
from dataclasses import replace

from iris.cluster.types import Entrypoint
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.identity_context_distill.runtime_recipe import xla_precision_env
from experiments.post_training.identity_path_b.ablation import AblationArm, export_worker, frozen_names
from experiments.post_training.identity_path_b.ablation_launch import owned_terminal, record, retire_native, submission
from experiments.post_training.identity_path_b.async_launch import checked_export_event, native_submit
from experiments.post_training.identity_path_b.checkpoints import TEMPORARY_PREFIX, CheckpointEvent, event_bytes
from experiments.post_training.identity_path_b.export_preflight import export_write_probe
from experiments.post_training.identity_path_b.worker import read_pinned
from experiments.post_training.identity_path_b.workflow import EXPORT_LIFETIME, EXPORT_RESOURCES

SOURCE_BIND = "admission/audits/source-bind.json"


def export_only_worker(*, training: dict, event: CheckpointEvent, recovery_root: str, evidence: dict) -> None:
    """Validate the original completed training, preserve metadata, then export once."""
    root = training["run_root"]
    if (
        event.arm != AblationArm.TRAIN_SPECIAL_ROWS
        or training["arm"] != event.arm
        or event.completed_updates != 36
        or event.training_pin != training["training_pin"]
        or event.packed_data_sha256 != training["packed_sha256"]
        or event.native_checkpoint != f"{root}/checkpoints/step-35"
        or event.hf_export != f"{root}/hf/updates-36"
        or event.eval_scratch != f"{root}/eval/updates-36"
        or not recovery_root.startswith(TEMPORARY_PREFIX)
        or recovery_root.startswith(root + "/")
    ):
        raise ValueError("Recovery must bind the original A1 final event and a separate TTL root")
    committed = read_pinned(f"{root}/final-checkpoint-event.json", evidence["event_sha256"])
    result = json.loads(read_pinned(f"{root}/training-result.json", evidence["training_result_sha256"]))
    metadata = json.loads(read_pinned(f"{event.native_checkpoint}/metadata.json", evidence["metadata_sha256"]))
    exposure = json.loads(read_pinned(training["exposure_path"], training["exposure_sha256"]))
    if (
        committed != event_bytes(event)
        or event.exposure != exposure["cumulative_batches"][-1]
        or result["arm"] != event.arm
        or result["completed_updates"] != 36
        or len(result["updates"]) != 36
        or result["training_pin"] != event.training_pin
        or result["packed_sha256"] != training["packed_sha256"]
        or result["exposure_sha256"] != training["exposure_sha256"]
        or result["special_rows_frozen"] is not False
        or set(result["fully_frozen_hf_names"]) != frozen_names(AblationArm.TRAIN_SPECIAL_ROWS)
        or metadata["step"] != 35
        or metadata["is_temporary"] is not True
    ):
        raise ValueError("Recovery evidence differs from the completed immutable A1 training")
    hf = StoragePath(event.hf_export)
    existing = {str(directory / name) for directory, _, names in hf.walk() for name in names}
    if existing != {str(hf / SOURCE_BIND)}:
        raise ValueError("Recovery refuses existing model/config/other partial export objects")
    source_bind = read_pinned(str(hf / SOURCE_BIND), evidence["source_bind_sha256"])
    if len(source_bind) != evidence["source_bind_bytes"]:
        raise ValueError("Original failed-export source receipt length differs")
    archive = StoragePath(recovery_root) / "preserved-source-bind.json"
    if archive.exists():
        raise FileExistsError(str(archive))
    archive.write_bytes(source_bind)
    if archive.read_bytes() != source_bind:
        raise ValueError("Failed-export source receipt preservation readback differs")
    # Only the probe uses a fresh namespace. The exporter receives the original event.
    probe_event = replace(event, eval_scratch=f"{recovery_root}/preflight")
    probe = export_write_probe(probe_event)
    record(f"{recovery_root}/export-preflight.json", probe)
    environment = {
        **xla_precision_env(json.loads(training["runtime_bytes"])["xla_precision"]),
        "JAX_PLATFORMS": "cuda,cpu",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
    }
    kwargs = {
        key: training[key] for key in ("run_id", "runtime_bytes", "runtime_sha256", "source_receipt", "source_objects")
    }
    export = native_submit(
        Entrypoint.from_callable(export_worker, event=event, **kwargs),
        name="export-36",
        resources=EXPORT_RESOURCES,
        replicas=1,
        env_vars=environment,
        lifetime=EXPORT_LIFETIME,
    )
    record(f"{recovery_root}/export-submitted.json", {"job": str(export.job_id), "event": event.payload()})
    owned_terminal(export, recovery_root, "export")
    checked_export_event(event)
    if (hf / SOURCE_BIND).read_bytes() != source_bind or archive.read_bytes() != source_bind:
        raise ValueError("Original failed-export source receipt changed; native retained")
    retire_native(event, training)
    record(
        f"{recovery_root}/bfcl-ready.json",
        {"status": "audited_HF_ready_BFCL_pending", "event": event.payload(), "retries": 0},
    )


def recovery_submission(
    *, name: str, training: dict, event: CheckpointEvent, recovery_root: str, evidence: dict, credentials: dict[str, str]
) -> dict:
    """Build one fresh CPU lifetime owner with the original admitted source settings."""
    if re.fullmatch(r"pathb-bfcl-a1-export-recovery-r1-[0-9a-f]{12}", name) is None:
        raise ValueError("Require a fresh A1 export-only recovery root name")
    # Reuse source/runtime/credential validation and the proven CPU request policy.
    request = submission(
        name=f"pathb-bfcl-a1-r1-{training['training_pin'][:12]}", training=training, credentials=credentials
    )
    request["name"] = name
    request["entrypoint"] = Entrypoint.from_callable(
        export_only_worker, training=training, event=event, recovery_root=recovery_root, evidence=evidence
    )
    return request
