# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The two owner-authorized recoveries after the RNO write suspension."""

import hashlib
import json
import math
import re
from pathlib import Path

from iris.cluster.types import Entrypoint
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.identity_context_distill.runtime_recipe import xla_precision_env
from experiments.post_training.identity_path_b.ablation import AblationArm, export_worker, frozen_names
from experiments.post_training.identity_path_b.ablation_launch import owned_terminal, record, submission
from experiments.post_training.identity_path_b.async_launch import checked_export_event, native_submit
from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent, event_bytes
from experiments.post_training.identity_path_b.export_preflight import export_write_probe
from experiments.post_training.identity_path_b.launch import SOURCE_RECEIPT
from experiments.post_training.identity_path_b.worker import read_pinned
from experiments.post_training.identity_path_b.workflow import EXPORT_LIFETIME, EXPORT_RESOURCES

BINDING_SHA256 = "0447b34b8d7b85de5d2e3fb692d647fc7c4d7ed1af8b0608b2b456d53fcdb757"


def binding(label: str) -> dict:
    raw = Path(__file__).with_name("write_recovery_binding.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != BINDING_SHA256:
        raise ValueError("Committed recovery execution identities changed")
    return json.loads(raw)[label]


def checkpoint_event(payload: dict) -> CheckpointEvent:
    return CheckpointEvent(**{key: payload[key] for key in CheckpointEvent.__dataclass_fields__})


def checked_original_training() -> tuple[dict, CheckpointEvent]:
    """Require the actual immutable final native and all72 finite update records."""
    item = binding("A9")
    evidence = item["original_evidence"]
    training = item["original_training"]
    event = checkpoint_event(item["original_event"])
    paths = {
        "event": training["run_root"] + "/final-checkpoint-event.json",
        "training_result": training["run_root"] + "/training-result.json",
        "metadata": event.native_checkpoint + "/metadata.json",
    }
    actual = {}
    for key, uri in paths.items():
        entry = evidence[key]
        if entry["uri"] != uri or re.fullmatch("[0-9a-f]{64}", entry["sha256"]) is None:
            raise ValueError("Require actual same-source original A9 evidence")
        raw = read_pinned(uri, entry["sha256"])
        if len(raw) != entry["bytes"]:
            raise ValueError("Original A9 evidence length differs")
        actual[key] = json.loads(raw)
    result = actual["training_result"]
    exposure = json.loads(read_pinned(training["exposure_path"], training["exposure_sha256"]))
    if (
        actual["event"] != event.payload()
        or event.exposure != exposure["cumulative_batches"][-1]
        or result["arm"] != "A9"
        or result["completed_updates"] != 72
        or result["training_pin"] != training["training_pin"]
        or result["packed_sha256"] != training["packed_sha256"]
        or result["exposure_sha256"] != training["exposure_sha256"]
        or result["special_rows_frozen"] is not True
        or set(result["fully_frozen_hf_names"]) != frozen_names(AblationArm.GLM_ON_TENTH_72)
        or [row["completed_updates"] for row in result["updates"]] != list(range(1, 73))
        or not all(math.isfinite(row["loss"]) for row in result["updates"])
        or actual["metadata"]["step"] != 71
        or actual["metadata"]["is_temporary"] is not True
    ):
        raise ValueError("Original A9 native/training evidence differs from the sealed science")
    return actual, event


def retire_recovered_native(event: CheckpointEvent) -> None:
    """Retire only original native71, after its fresh export's full storage audit."""
    item = binding("A9")
    if event != checkpoint_event(item["event"]):
        raise ValueError("Retirement must bind the exact new export and original native71")
    checked_export_event(event)
    audit = json.loads((StoragePath(event.hf_export) / "path-b-storage-audit.json").read_bytes())
    expected = frozen_names(AblationArm.GLM_ON_TENTH_72)
    if (
        audit["source_receipt_sha256"] != SOURCE_RECEIPT["sha256"]
        or audit["schema"] != "path_b_storage_v1"
        or audit["frozen_native_tensors"] != len(expected)
        or audit["trained_master_cast_tensors"] != 502 - len(expected)
        or len(audit["tensors"]) != 502
        or {name for name, row in audit["tensors"].items() if row["frozen_source"]} != expected
    ):
        raise ValueError("Retirement requires the unchanged all502 source/cast audit")
    checked_original_training()
    native = StoragePath(event.native_checkpoint)
    native.rmtree()
    if native.exists():
        raise RuntimeError("Audited recovered native retirement did not finish")
    (StoragePath(item["root"]) / "native-retired.json").write_bytes(event_bytes(event))


def export_only_worker(*, runtime_bytes: bytes) -> None:
    """Own exactly one original exporter child; never launch training or reuse failed HF."""
    item = binding("A9")
    checked_original_training()
    root = item["root"]
    event = checkpoint_event(item["event"])
    if StoragePath(root).exists():
        raise FileExistsError("Recovery output already exists; no automatic retry")
    # This tiny CPU probe precedes the only GPU allocation.
    probe = export_write_probe(event)
    record(root + "/workflow/export-preflight.json", probe)
    (StoragePath(root) / "final-checkpoint-event.json").write_bytes(event_bytes(event))
    record(root + "/original-training-evidence.json", item["original_evidence"])
    training = {**item["training"], "runtime_bytes": runtime_bytes}
    environment = {
        **xla_precision_env(json.loads(runtime_bytes)["xla_precision"]),
        "JAX_PLATFORMS": "cuda,cpu",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
    }
    kwargs = {
        key: training[key] for key in ("run_id", "runtime_bytes", "runtime_sha256", "source_receipt", "source_objects")
    }
    export = native_submit(
        Entrypoint.from_callable(export_worker, event=event, **kwargs),
        name="export-72",
        resources=EXPORT_RESOURCES,
        replicas=1,
        env_vars=environment,
        lifetime=EXPORT_LIFETIME,
    )
    record(root + "/workflow/export-submitted.json", {"job": str(export.job_id), "event": event.payload()})
    owned_terminal(export, root, "export")
    retire_recovered_native(event)
    record(
        root + "/bfcl-ready.json", {"status": "audited_HF_ready_BFCL_pending", "event": event.payload(), "retries": 0}
    )


def recovery_submission(*, label: str, runtime_bytes: bytes, credentials: dict[str, str]) -> dict:
    """Build fresh recovery roots with the original scientific workers and policy."""
    item = binding(label)
    training = {**item["training"], "runtime_bytes": runtime_bytes}
    request = submission(
        name=f"pathb-bfcl-{label.lower()}-r1-{training['training_pin'][:12]}",
        training=training,
        credentials=credentials,
    )
    request["name"] = item["job_name"]
    if label == "A9":
        request["entrypoint"] = Entrypoint.from_callable(export_only_worker, runtime_bytes=runtime_bytes)
    return request
