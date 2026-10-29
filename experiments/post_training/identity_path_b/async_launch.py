# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-task checkpoint children using native Iris auth, workspace and environment.

The only training-side work is an immutable event write plus a bounded native
submit. Export and evaluation completion are independent of training. Native
Iris RPC transport recovery is unchanged; there are no application resubmits or
job failure/preemption retries.
"""

import hashlib
import json
import os
from dataclasses import dataclass, replace

import jax
from fray.iris_backend import convert_constraints, convert_resources, resolve_coscheduling
from fray.types import ResourceConfig
from iris.client.client import iris_ctx, iris_ctx_scope
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import SecretBehavior
from rigging.timing import Duration

from experiments.post_training.identity_context_distill.lr0_probe import RESOURCES
from experiments.post_training.identity_context_distill.runtime_recipe import xla_precision_env
from experiments.post_training.identity_path_b.checkpoints import AsyncCheckpoints, CheckpointEvent
from experiments.post_training.identity_path_b.worker import export_worker

CHILD_LIFETIME = 12 * 3600
TREND_LIFETIME = 24 * 3600
CPU_RESOURCES = ResourceConfig.with_cpu(cpu=8, ram="32g", disk="50g", target_cluster="cw-rno2a")


def write_json(path: str, payload: dict) -> None:
    StoragePath(path).write_bytes((json.dumps(payload, sort_keys=True, indent=2) + "\n").encode())


def regional_task_context():
    """Use the assigned controller/task token, as the OPD nested launcher does."""
    info = get_job_info()
    if info is None or not info.controller_address or info.controller_address != os.environ.get("IRIS_CONTROLLER_URL"):
        raise RuntimeError("Native task controller assignment is missing or inconsistent")
    # Native iris_ctx creates IrisClient.in_cluster(controller_address,
    # bundle_id=info.bundle_id), whose credentials come from info.credentials.
    # This is the OPD ambient-client route, with the existing bundle reused.
    context = iris_ctx()
    if context.client is None or context.job_id != info.job_id:
        raise RuntimeError("Independent children require the assigned native Iris task context")
    return context


def native_submit(
    entrypoint: Entrypoint, *, name: str, resources: ResourceConfig, replicas: int, env_vars: dict, lifetime: int
):
    """Submit once through the task's authenticated context; never open host auth."""
    context = regional_task_context()
    assert context.client is not None
    # OPD ef91feee skyrl.py:809 clears target_cluster for an ambient task.
    # The root is placed through marin's federation; an in-region child uses
    # the assigned peer controller and inherits its location, not federation.
    resources = replace(resources, target_cluster=None)
    return context.client.submit(
        entrypoint=entrypoint,
        name=name,
        resources=convert_resources(resources),
        constraints=convert_constraints(resources) or None,
        coscheduling=resolve_coscheduling(resources, replicas),
        replicas=replicas,
        ports=["jax"] if replicas == 4 else [],
        # No extras/setup override: native Iris inherits the parent's resolved
        # setup and env, and forwards its already-authorized storage secrets.
        environment=EnvironmentSpec(env_vars=env_vars, secret_behavior=SecretBehavior.INHERIT),
        timeout=Duration.from_seconds(lifetime),
        max_retries_failure=0,
        max_retries_preemption=0,
        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
    )


@dataclass(frozen=True)
class TrendInputs:
    identity_path: str
    controls_path: str


@dataclass(frozen=True)
class ExportBinding:
    run_id: str
    runtime_bytes: bytes
    runtime_sha256: str
    source_receipt: dict
    source_objects: dict
    trend_inputs: TrendInputs


def checked_export_event(event: CheckpointEvent) -> bytes:
    """An audited export's final event must equal the committed native event."""
    raw = (StoragePath(event.hf_export) / "path-b-checkpoint-event.json").read_bytes()
    if json.loads(raw) != event.payload():
        raise ValueError("Audited export completion event differs from the committed training event")
    return raw


def export_then_submit_trend(event: CheckpointEvent, binding: ExportBinding) -> None:
    """All4 export ranks run the cleared worker; only process0 submits the eval."""
    try:
        export_worker(
            event=event,
            run_id=binding.run_id,
            runtime_bytes=binding.runtime_bytes,
            runtime_sha256=binding.runtime_sha256,
            source_receipt=binding.source_receipt,
            source_objects=binding.source_objects,
        )
    except Exception as error:
        context = regional_task_context()
        assert context.task_attempt is not None
        rank = context.task_attempt.task_id.require_task()[1]
        write_json(
            f"{event.eval_scratch}/export-failed-task-{rank}.json",
            {
                "status": "export_failed",
                "event": event.payload(),
                "error_type": type(error).__name__,
                "application_resubmits": 0,
            },
        )
        raise
    if jax.process_index() != 0:
        return
    checked_export_event(event)
    # Import here prevents a cycle: trend's evaluator reuses this module's
    # native submission policy while this callback binds its actual callable.
    from experiments.post_training.identity_path_b.trend import trend_worker  # noqa: PLC0415

    try:
        job = native_submit(
            Entrypoint.from_callable(trend_worker, args=(event, binding.trend_inputs)),
            name=f"trend-{event.completed_updates:06d}",
            resources=CPU_RESOURCES,
            replicas=1,
            env_vars={"JAX_PLATFORMS": "cpu"},
            lifetime=TREND_LIFETIME,
        )
        write_json(f"{event.eval_scratch}/submission.json", {"stage": "eval_submitted", "job": str(job.job_id)})
    except Exception as error:
        write_json(
            f"{event.eval_scratch}/submission.json",
            {
                "stage": "eval_submit_failed",
                "error_type": type(error).__name__,
                "application_resubmits": 0,
            },
        )
        raise


@dataclass(frozen=True)
class CheckpointHookFactory:
    run_root: str
    binding: ExportBinding

    def __call__(self) -> AsyncCheckpoints:
        # ContextVar state does not automatically propagate into the background
        # submit thread. Capture the authenticated context on the worker thread,
        # then explicitly scope it for submission. Only plain binding data was
        # serialized into the root Entrypoint, never a client/token.
        context = regional_task_context()

        def submit(payload: dict) -> dict:
            event = CheckpointEvent(
                **{key: value for key, value in payload.items() if key in CheckpointEvent.__dataclass_fields__}
            )
            with iris_ctx_scope(context):
                job = native_submit(
                    Entrypoint.from_callable(export_then_submit_trend, args=(event, self.binding)),
                    name=f"export-{event.completed_updates:06d}",
                    resources=RESOURCES,
                    replicas=4,
                    env_vars=xla_precision_env(json.loads(self.binding.runtime_bytes)["xla_precision"]),
                    lifetime=CHILD_LIFETIME,
                )
            return {"export": str(job.job_id), "eval": "submitted by export writer after audited completion"}

        def persist(name: str, raw: bytes) -> None:
            StoragePath(f"{self.run_root}/checkpoint-events/{name}.json").write_bytes(raw)
            payload = json.loads(raw)
            write_json(
                f"{payload['eval_scratch']}/trend-pending.json",
                {
                    "status": "pending_export",
                    "event": payload,
                    "event_sha256": hashlib.sha256(raw).hexdigest(),
                    "own_profile": "UNGRADED_PENDING_ROLE_AWARE_SEMANTIC_READ",
                },
            )

        return AsyncCheckpoints(
            write_event=persist,
            submit_export_eval=submit,
            write_submission_result=lambda name, result: write_json(
                f"{self.run_root}/checkpoint-events/{name}-submission.json", result
            ),
        )
