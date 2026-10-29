# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU lifetime owner for sibling training, export and diagnostic jobs.

Only this task waits on descendants. Finished GPU tasks return immediately.
Diagnostic failures never authorize a stop of ongoing scientific training.
"""

import hashlib
import json
import logging
import threading
import time
from queue import Queue

from fray.current_client import current_client
from fray.types import ActorConfig
from iris.client.client import iris_ctx_scope
from iris.cluster.types import Entrypoint
from iris.resources.state import JobState
from rigging.timing import ExponentialBackoff

from experiments.post_training.identity_context_distill.lr0_probe import RESOURCES
from experiments.post_training.identity_context_distill.runtime_recipe import xla_precision_env
from experiments.post_training.identity_path_b.async_launch import (
    CHILD_LIFETIME,
    CPU_RESOURCES,
    TREND_LIFETIME,
    CheckpointHookFactory,
    ExportBinding,
    checked_export_event,
    export_checkpoint,
    native_submit,
    regional_task_context,
    write_json,
)
from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent, event_bytes
from experiments.post_training.identity_path_b.worker import read_pinned, train_worker

logger = logging.getLogger(__name__)
# Root lifetime must cover late export + trend after the training allocation is
# released. It is a CPU allocation, not an idle training/export GPU allocation.
ROOT_LIFETIME = CHILD_LIFETIME + CHILD_LIFETIME + TREND_LIFETIME + 3600


class CheckpointInbox:
    """Short native-actor enqueue, with immutable event identity and no I/O."""

    def __init__(self, messages: Queue, expected: dict, checkpoint_updates: tuple[int, ...]):
        self.messages = messages
        self.expected = expected
        self.checkpoint_updates = checkpoint_updates
        self.seen: dict[int, bytes] = {}
        self.lock = threading.Lock()

    def enqueue(self, payload: dict) -> dict:
        event = CheckpointEvent(**{k: payload[k] for k in CheckpointEvent.__dataclass_fields__})
        if payload != event.payload() or any(payload[k] != value for k, value in self.expected.items()):
            raise ValueError("Checkpoint event differs from the admitted training binding")
        if event.completed_updates not in self.checkpoint_updates:
            raise ValueError("Checkpoint event is outside the frozen cadence")
        raw = event_bytes(event)
        with self.lock:
            previous = self.seen.get(event.completed_updates)
            if previous is not None and previous != raw:
                raise ValueError("Conflicting committed event at the same update")
            if previous is None:
                self.seen[event.completed_updates] = raw
                self.messages.put(("checkpoint", event))
        return {"cpu_root_enqueued": True, "event_sha256": hashlib.sha256(raw).hexdigest()}


def observe_terminal(job, stage: str, messages: Queue) -> None:
    """Native owned-job waiting, retaining ownership across observation errors.

    A read/transport exception does not mean the job stopped. Keep the handle
    active and continue native lifecycle observation; never resubmit/cancel a
    scientific job because a diagnostic read failed. The CPU task's explicit
    finite lifetime remains the outer execution bound.
    """
    backoff = ExponentialBackoff(initial=1, maximum=30, factor=2, jitter=0)
    while True:
        try:
            status = job.wait(timeout=ROOT_LIFETIME, raise_on_failure=False)
        except Exception as error:
            messages.put(("observation_error", (str(job.job_id), stage, type(error).__name__)))
            time.sleep(backoff.next_interval())
            continue
        messages.put(("terminal", (str(job.job_id), stage, status.state)))
        return


class Workflow:
    """Lifecycle state at the native job boundary, shared by root and CPU tests."""

    def __init__(self, run_root: str, binding: ExportBinding, messages: Queue):
        self.run_root, self.binding, self.messages = run_root, binding, messages
        self.active: dict[str, tuple[str, CheckpointEvent | None]] = {}
        self.records: dict[str, dict] = {}
        self.received: set[int] = set()
        self.training_state: JobState | None = None

    def record(self, key: str, payload: dict) -> None:
        self.records[key] = payload
        # Receipt/storage failure must not unwind the CPU lifetime owner and
        # cascade-kill still-active scientific work. Retain/log the failure.
        try:
            write_json(f"{self.run_root}/workflow/{key}.json", payload)
        except Exception as error:
            logger.error("Workflow receipt %s failed: %s", key, type(error).__name__)

    def checkpoint_record(self, event: CheckpointEvent, status: str, **details) -> None:
        payload = {"status": status, "event": event.payload(), **details}
        self.record(f"checkpoint-{event.completed_updates:06d}", payload)
        try:
            write_json(f"{event.eval_scratch}/workflow-status.json", payload)
        except Exception as error:
            logger.error("Checkpoint diagnostic status write failed: %s", type(error).__name__)

    def own(self, job, stage: str, event: CheckpointEvent | None = None) -> None:
        self.active[str(job.job_id)] = (stage, event)
        threading.Thread(target=observe_terminal, args=(job, stage, self.messages), daemon=True).start()

    def checkpoint(self, event: CheckpointEvent) -> None:
        self.received.add(event.completed_updates)
        try:
            job = native_submit(
                Entrypoint.from_callable(export_checkpoint, event, self.binding),
                name=f"export-{event.completed_updates:06d}",
                resources=RESOURCES,
                replicas=4,
                env_vars={
                    **xla_precision_env(json.loads(self.binding.runtime_bytes)["xla_precision"]),
                    "JAX_PLATFORMS": "cuda,cpu",
                },
                lifetime=CHILD_LIFETIME,
            )
        except Exception as error:
            self.checkpoint_record(
                event, "export_submit_failed", error_type=type(error).__name__, application_resubmits=0
            )
            return
        self.own(job, "export", event)
        self.checkpoint_record(event, "export_submitted", job=str(job.job_id))

    def terminal(self, job_id: str, stage: str, state: JobState) -> None:
        _, event = self.active.pop(job_id)
        self.record(job_id.rsplit("/", 1)[-1], {"stage": stage, "state": str(state), "job": job_id})
        if stage == "train":
            self.training_state = state
            return
        assert event is not None
        self.checkpoint_record(event, f"{stage}_{state}", job=job_id)
        if stage != "export" or state != JobState.SUCCEEDED:
            return
        assert event is not None
        try:
            checked_export_event(event)
            from experiments.post_training.identity_path_b.trend import trend_worker  # noqa: PLC0415

            job = native_submit(
                Entrypoint.from_callable(trend_worker, event, self.binding.trend_inputs),
                name=f"trend-{event.completed_updates:06d}",
                resources=CPU_RESOURCES,
                replicas=1,
                env_vars={"JAX_PLATFORMS": "cpu"},
                lifetime=TREND_LIFETIME,
            )
        except Exception as error:
            self.checkpoint_record(
                event,
                "export_admission_or_trend_submit_failed",
                error_type=type(error).__name__,
                application_resubmits=0,
            )
            return
        self.own(job, "trend", event)
        self.checkpoint_record(event, "trend_submitted", job=str(job.job_id))

    def run(self) -> None:
        while self.active:
            kind, value = self.messages.get()
            if kind == "checkpoint":
                self.checkpoint(value)
            elif kind == "terminal":
                self.terminal(*value)
            else:
                job, stage, error = value
                self.record(
                    f"observation-{job.rsplit('/', 1)[-1]}",
                    {
                        "status": "observation_failed_job_still_owned",
                        "job": job,
                        "stage": stage,
                        "error_type": error,
                        "application_resubmits": 0,
                    },
                )


def workflow_worker(*, training: dict, binding: ExportBinding) -> None:
    """One native CPU root; all three scientific stages are direct siblings."""
    context = regional_task_context()
    if "checkpoint_hook_factory" in training:
        raise ValueError("Only the CPU lifetime owner may bind checkpoint callbacks")
    if any(
        training[k] != getattr(binding, k)
        for k in ("run_id", "runtime_bytes", "runtime_sha256", "source_receipt", "source_objects")
    ):
        raise ValueError("Train/export source and runtime bindings must match")
    exposure = json.loads(read_pinned(training["exposure_path"], training["exposure_sha256"]))
    updates = tuple(exposure["checkpoint_updates"])
    expected = {k: training[k] for k in ("arm", "training_pin")}
    expected["packed_data_sha256"] = training["packed_sha256"]
    messages: Queue = Queue()
    actor_name = "checkpoint-inbox"
    hosted = current_client().host_actor(
        CheckpointInbox,
        messages,
        expected,
        updates,
        name=actor_name,
        actor_config=ActorConfig(max_concurrency=1, max_restarts=0, max_task_retries=0),
    )
    workflow = Workflow(training["run_root"], binding, messages)
    try:
        kwargs = dict(
            training,
            checkpoint_hook_factory=CheckpointHookFactory(training["run_root"], f"{context.job_id}/{actor_name}-0"),
        )
        # Explicit CPU-root context makes all submissions siblings. No task
        # inherits a context captured in a finished GPU parent.
        with iris_ctx_scope(context):
            job = native_submit(
                Entrypoint.from_callable(train_worker, **kwargs),
                name="train",
                resources=RESOURCES,
                replicas=4,
                env_vars={
                    **xla_precision_env(json.loads(binding.runtime_bytes)["xla_precision"]),
                    "JAX_PLATFORMS": "cuda,cpu",
                },
                lifetime=CHILD_LIFETIME,
            )
            workflow.own(job, "train")
            workflow.run()
        workflow.record(
            "final",
            {
                "training_state": str(workflow.training_state),
                "missing_checkpoint_events": sorted(set(updates) - workflow.received),
                "diagnostic_status": "see individual complete/failed/incomplete records; no scientific stop implied",
                "job_and_coverage_records": dict(workflow.records),
            },
        )
        if workflow.training_state != JobState.SUCCEEDED:
            raise RuntimeError("Scientific training failed; all owned diagnostic jobs have reached terminal state")
    finally:
        hosted.shutdown()
