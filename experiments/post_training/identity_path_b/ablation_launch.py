# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Five independent CPU owners: train, preflight, audited export, then BFCL handoff."""

import hashlib
import json
import re
import threading
from queue import Queue

from fray.iris_backend import convert_constraints, convert_resources
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.resources.state import JobState
from iris.rpc import job_pb2
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import SecretBehavior
from rigging.timing import Duration

from experiments.post_training.identity_context_distill.lr0_probe import RESOURCES
from experiments.post_training.identity_context_distill.runtime_recipe import xla_precision_env
from experiments.post_training.identity_path_b.ablation import (
    AblationArm,
    export_worker,
    frozen_names,
    train_worker,
    updates_for,
)
from experiments.post_training.identity_path_b.async_launch import (
    CPU_RESOURCES,
    checked_export_event,
    native_submit,
    write_optional,
)
from experiments.post_training.identity_path_b.checkpoints import TEMPORARY_PREFIX, CheckpointEvent, event_bytes
from experiments.post_training.identity_path_b.export_preflight import export_write_probe
from experiments.post_training.identity_path_b.launch import RUNTIME_SHA256, SOURCE_OBJECTS, SOURCE_RECEIPT
from experiments.post_training.identity_path_b.worker import read_pinned
from experiments.post_training.identity_path_b.workflow import (
    EXPORT_LIFETIME,
    EXPORT_RESOURCES,
    ROOT_LIFETIME,
    observe_terminal,
)


def record(path: str, payload: dict) -> None:
    write_optional(path, (json.dumps(payload, indent=2) + "\n").encode())


def owned_terminal(job, root: str, stage: str):
    messages = Queue()
    thread = threading.Thread(target=observe_terminal, args=(job, stage, messages), daemon=True)
    thread.start()
    while True:
        kind, value = messages.get()
        if kind == "terminal":
            thread.join()
            _, _, state = value
            record(f"{root}/workflow/{stage}-terminal.json", {"job": str(job.job_id), "state": str(state)})
            if state != JobState.SUCCEEDED:
                raise RuntimeError(f"Owned ablation {stage} terminated {state}")
            return
        job_name, _, error = value
        record(f"{root}/workflow/{stage}-observation-error.json", {"job": job_name, "error_type": error})


def retire_native(event: CheckpointEvent, training: dict) -> None:
    """Only after terminal audited export, delete this exact temporary final native."""
    root = training["run_root"]
    updates = updates_for(AblationArm(training["arm"]))
    if not root.startswith(TEMPORARY_PREFIX) or "/bfcl-ablation-" not in root:
        raise ValueError("Retirement is confined to the new ablation namespace")
    if (
        event.native_checkpoint != f"{root}/checkpoints/step-{updates - 1}"
        or event.hf_export != f"{root}/hf/updates-{updates}"
    ):
        raise ValueError("Retirement differs from this exact bound final checkpoint")
    checked_export_event(event)
    audit = json.loads((StoragePath(event.hf_export) / "path-b-storage-audit.json").read_bytes())
    expected = frozen_names(AblationArm(event.arm))
    if (
        audit["source_receipt_sha256"] != SOURCE_RECEIPT["sha256"]
        or audit["schema"] != "path_b_storage_v1"
        or audit["frozen_native_tensors"] != len(expected)
        or audit["trained_master_cast_tensors"] != 502 - len(expected)
        or len(audit["tensors"]) != 502
        or {name for name, row in audit["tensors"].items() if row["frozen_source"]} != expected
    ):
        raise ValueError("Retirement requires all502 exact arm-specific cast/source audits")
    path = StoragePath(event.native_checkpoint)
    metadata = json.loads((path / "metadata.json").read_bytes())
    if metadata["step"] != updates - 1 or metadata["is_temporary"] is not True:
        raise ValueError("Retirement requires the committed temporary final native metadata")
    path.rmtree()
    if path.exists():
        raise RuntimeError("Audited final native retirement did not finish")
    (StoragePath(root) / "native-retired.json").write_bytes(event_bytes(event))


def workflow_worker(*, training: dict) -> None:
    root = training["run_root"]
    updates = updates_for(AblationArm(training["arm"]))
    runtime = json.loads(training["runtime_bytes"])
    environment = {
        **xla_precision_env(runtime["xla_precision"]),
        "JAX_PLATFORMS": "cuda,cpu",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
    }
    train = native_submit(
        Entrypoint.from_callable(train_worker, **training),
        name="train",
        resources=RESOURCES,
        replicas=4,
        env_vars=environment,
        lifetime=12 * 3600,
    )
    record(f"{root}/workflow/train-submitted.json", {"job": str(train.job_id)})
    owned_terminal(train, root, "train")
    payload = json.loads((StoragePath(root) / "final-checkpoint-event.json").read_bytes())
    event = CheckpointEvent(**{key: payload[key] for key in CheckpointEvent.__dataclass_fields__})
    exposure = json.loads(read_pinned(training["exposure_path"], training["exposure_sha256"]))
    if (
        payload != event.payload()
        or event.arm != training["arm"]
        or event.completed_updates != updates
        or event.training_pin != training["training_pin"]
        or event.packed_data_sha256 != training["packed_sha256"]
        or event.exposure != exposure["cumulative_batches"][-1]
        or event.native_checkpoint != f"{root}/checkpoints/step-{updates - 1}"
        or event.hf_export != f"{root}/hf/updates-{updates}"
        or event.eval_scratch != f"{root}/eval/updates-{updates}"
    ):
        raise ValueError("Committed final event differs from the exact admitted ablation")
    probe = export_write_probe(event)
    record(f"{root}/workflow/export-preflight.json", probe)
    kwargs = {
        key: training[key] for key in ("run_id", "runtime_bytes", "runtime_sha256", "source_receipt", "source_objects")
    }
    export = native_submit(
        Entrypoint.from_callable(export_worker, event=event, **kwargs),
        name=f"export-{updates}",
        resources=EXPORT_RESOURCES,
        replicas=1,
        env_vars=environment,
        lifetime=EXPORT_LIFETIME,
    )
    record(f"{root}/workflow/export-submitted.json", {"job": str(export.job_id)})
    owned_terminal(export, root, "export")
    checked_export_event(event)
    retire_native(event, training)
    record(
        f"{root}/bfcl-ready.json",
        {
            "status": "audited_HF_ready_BFCL_pending",
            "event": event.payload(),
            "BFCL": "executor uses unchanged registered caba0fa method/reference",
            "retries": 0,
        },
    )


def submission(*, name: str, training: dict, credentials: dict[str, str]) -> dict:
    arm = AblationArm(training["arm"])
    if re.fullmatch(f"pathb-bfcl-{arm.lower()}-r1-[0-9a-f]{{12}}", name) is None:
        raise ValueError("Require the arm's exact fresh ablation root name")
    if re.fullmatch("[0-9a-f]{40}", training["training_pin"]) is None:
        raise ValueError("Require the immutable full training pin")
    if any(not training[key].startswith(TEMPORARY_PREFIX) for key in ("run_root", "packed_path", "exposure_path")):
        raise ValueError("Ablation inputs and outputs require RNO14d")
    if (
        training["source_receipt"] != SOURCE_RECEIPT
        or training["source_objects"] != SOURCE_OBJECTS
        or training["runtime_sha256"] != RUNTIME_SHA256
        or hashlib.sha256(training["runtime_bytes"]).hexdigest() != RUNTIME_SHA256
    ):
        raise ValueError("Ablation source/precision/runtime pins differ")
    secrets = {key: credentials[key] for key in ("CW_KEY_ID", "CW_KEY_SECRET")}
    secrets.update(AWS_ACCESS_KEY_ID=secrets["CW_KEY_ID"], AWS_SECRET_ACCESS_KEY=secrets["CW_KEY_SECRET"])
    return dict(
        name=name,
        entrypoint=Entrypoint.from_callable(workflow_worker, training=training),
        resources=convert_resources(CPU_RESOURCES),
        constraints=convert_constraints(CPU_RESOURCES) or None,
        replicas=1,
        environment=EnvironmentSpec(
            extras=["gpu"],
            secrets=secrets,
            secret_behavior=SecretBehavior.ISOLATE,
            env_vars={
                **xla_precision_env(json.loads(training["runtime_bytes"])["xla_precision"]),
                "JAX_PLATFORMS": "cpu",
            },
        ),
        timeout=Duration.from_seconds(ROOT_LIFETIME),
        max_retries_failure=0,
        max_retries_preemption=0,
        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
    )
