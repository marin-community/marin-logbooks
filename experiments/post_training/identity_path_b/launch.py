# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Concrete CPU-root submission construction; execution belongs to the executor."""

import hashlib
import json
import re

from fray.iris_backend import convert_constraints, convert_resources
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from rigging.secrets import SecretBehavior
from rigging.timing import Duration

from experiments.post_training.identity_context_distill.runtime_recipe import xla_precision_env
from experiments.post_training.identity_path_b.async_launch import CPU_RESOURCES, ExportBinding
from experiments.post_training.identity_path_b.checkpoints import TEMPORARY_PREFIX
from experiments.post_training.identity_path_b.workflow import ROOT_LIFETIME, workflow_worker

RUNTIME_SHA256 = "565cf0b2e09b0e1d58b647b698364a04d41b80615b47f8e6e1dbc7167f0d8e9e"
SOURCE_RECEIPT = {
    "path": "s3://marin-rno2a/marin/experiments/snowball-identity/t310/lr0-r8/audits/source-tensor-hashes.json",
    "sha256": "af484f49ac2c1d67989495475bcc79bf7519d09281c8087321d906b67139fa97",
}
SOURCE_OBJECTS = {
    "path": "s3://marin-rno2a/marin/experiments/snowball-identity/t310/lr0-r8/audits/source-object-list.json",
    "sha256": "5b85b32675299015ca92921053d8aa2926cb007eb2a6a3cc2cf2cce3573603f1",
}


def submission(*, name: str, training: dict, binding: ExportBinding, credentials: dict[str, str]) -> dict:
    """Build one native RNO CPU lifetime owner; never submit or discover capacity."""
    if not re.fullmatch(r"pathb-bbase-(?:10|25)-[a-z0-9-]+", name):
        raise ValueError("Require a fresh unambiguous B-base arm root name")
    if training["arm"] not in {"10", "25"} or not name.startswith(f"pathb-bbase-{training['arm']}-"):
        raise ValueError("Root name and arm differ")
    if re.fullmatch(r"[0-9a-f]{40}", training["training_pin"]) is None:
        raise ValueError("Root requires an immutable full training revision")
    for key in ("run_root", "packed_path", "exposure_path"):
        if not training[key].startswith(TEMPORARY_PREFIX):
            raise ValueError("B-base root inputs/outputs require the RNO14-day TTL prefix")
    for path in (binding.trend_inputs.identity_path, binding.trend_inputs.controls_path):
        if not path.startswith(TEMPORARY_PREFIX):
            raise ValueError("Staged diagnostic inputs require the RNO14-day TTL prefix")
    if (
        training["runtime_bytes"] != binding.runtime_bytes
        or training["runtime_sha256"] != RUNTIME_SHA256
        or binding.runtime_sha256 != RUNTIME_SHA256
        or hashlib.sha256(binding.runtime_bytes).hexdigest() != RUNTIME_SHA256
    ):
        raise ValueError("Root requires the exact inherited numerical runtime receipt")
    if (
        training["source_receipt"] != SOURCE_RECEIPT
        or binding.source_receipt != SOURCE_RECEIPT
        or training["source_objects"] != SOURCE_OBJECTS
        or binding.source_objects != SOURCE_OBJECTS
    ):
        raise ValueError("Root source audit/object-list bindings differ from pinned v0.1")
    if training["run_id"] != binding.run_id or "checkpoint_hook_factory" in training:
        raise ValueError("Only the CPU workflow may bind checkpoint callbacks")
    # Secrets remain internal to the executor's authenticated submit. They are
    # never embedded in a callable, launch plan, command argument or receipt.
    secrets = {key: credentials[key] for key in ("CW_KEY_ID", "CW_KEY_SECRET")}
    secrets.update(AWS_ACCESS_KEY_ID=secrets["CW_KEY_ID"], AWS_SECRET_ACCESS_KEY=secrets["CW_KEY_SECRET"])
    return {
        "name": name,
        "entrypoint": Entrypoint.from_callable(workflow_worker, training=training, binding=binding),
        "resources": convert_resources(CPU_RESOURCES),
        "constraints": convert_constraints(CPU_RESOURCES) or None,
        "replicas": 1,
        "environment": EnvironmentSpec(
            extras=["gpu"],
            secrets=secrets,
            secret_behavior=SecretBehavior.ISOLATE,
            env_vars={**xla_precision_env(json.loads(binding.runtime_bytes)["xla_precision"]), "JAX_PLATFORMS": "cpu"},
        ),
        "timeout": Duration.from_seconds(ROOT_LIFETIME),
        "max_retries_failure": 0,
        "max_retries_preemption": 0,
        "priority_band": job_pb2.PRIORITY_BAND_INTERACTIVE,
    }
