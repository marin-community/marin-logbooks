# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
import threading
from concurrent.futures import Future
from dataclasses import replace
from queue import Queue
from types import SimpleNamespace

import pytest
from fray.actor import HostedActor
from iris.client.client import IrisContext, iris_ctx_scope
from iris.cluster.types import JobName
from iris.resources.state import JobState
from iris.rpc import job_pb2
from iris.testing.journeys.world import journey_world

from experiments.post_training.identity_path_b import async_launch, workflow
from experiments.post_training.identity_path_b.async_launch import ExportBinding, TrendInputs
from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent


def sample_event(completed=25):
    root = "s3://marin-rno2a/tmp/ttl=14d/path-b/fixture"
    return CheckpointEvent(
        completed,
        f"{root}/checkpoint/step-{completed-1}",
        f"{root}/hf/{completed}",
        f"{root}/eval/{completed}",
        {},
        "a" * 40,
        "b" * 64,
        "identity10",
    )


class ExecutionBoundary:
    """Fake external compute; lifecycle decisions use the real Iris controller."""

    def __init__(self, world, root):
        self.world, self.root = world, root
        self.changed = threading.Condition()
        self.jobs = {}

    def submit(self, entrypoint, *, name, replicas, **kwargs):
        # Exercise the native callable deserialization and actual signatures,
        # including keyword-only train_worker; no args=/kwargs= wrappers.
        fn, args, call_kwargs = entrypoint.resolve()
        inspect.signature(fn).bind(*args, **call_kwargs)
        with self.changed:
            ref = self.world.submit_child(self.root, name, tasks=replicas)
            self.world.settle()
            future = Future()
            job = SimpleNamespace(
                job_id=ref.wire_id,
                ref=ref,
                kwargs=call_kwargs,
                submit_options=kwargs,
                wait=lambda **unused: future.result(timeout=30),
                future=future,
            )
            self.jobs[name] = job
            self.changed.notify_all()
            return job

    def get(self, name):
        with self.changed:
            assert self.changed.wait_for(lambda: name in self.jobs, timeout=10), name
            return self.jobs[name]

    def finish(self, name, *, success=True):
        with self.changed:
            job = self.jobs[name]
            if success:
                self.world.succeed_all(job.ref)
            else:
                self.world.fail(job.ref[0])
            self.world.settle()
            state = self.world.job(job.ref).state
            terminal = JobState.SUCCEEDED if state == job_pb2.JOB_STATE_SUCCEEDED else JobState.FAILED
            job.future.set_result(SimpleNamespace(state=terminal))
            return job


def test_cpu_root_outlives_sibling_diagnostics_and_diagnostic_failure_does_not_stop_train(monkeypatch, tmp_path):
    receipts = {}
    monkeypatch.setattr(workflow, "export_write_probe", lambda event: {"bytes": 32})
    monkeypatch.setattr(workflow, "delete_audited_checkpoint", lambda *args: None)
    monkeypatch.setattr(workflow, "write_json", lambda path, value: receipts.update({path: value}))

    # Receipt storage is external; a valid export completion is exercised by
    # checked_export_event against the real event bytes below.
    def local_storage(path):
        result = tmp_path / "storage" / path.removeprefix("s3://")
        result.parent.mkdir(parents=True, exist_ok=True)
        return result

    monkeypatch.setattr(async_launch, "StoragePath", local_storage)
    root_events = Queue()
    hosted_stopped = threading.Event()

    class HostingBoundary:
        def host_actor(self, actor_class, messages, expected, updates, **unused):
            self.inbox = actor_class(messages, expected, updates)
            root_events.put(self.inbox)
            return HostedActor(SimpleNamespace(), hosted_stopped.set)

    hosting = HostingBoundary()
    monkeypatch.setattr(workflow, "current_client", lambda: hosting)
    monkeypatch.setattr(workflow, "read_pinned", lambda *unused: json.dumps({"checkpoint_updates": [25, 50]}).encode())
    context = IrisContext(job_id=JobName.from_wire("/journey/pathb-cpu-root"), client=None)
    monkeypatch.setattr(workflow, "regional_task_context", lambda: context)
    binding = ExportBinding(
        "fixture", b'{"xla_precision":"no_excess_precision"}', "c" * 64, {}, {}, TrendInputs("identity", "controls")
    )
    training = dict(
        run_root="s3://marin-rno2a/tmp/ttl=14d/path-b/fixture",
        run_id="fixture",
        arm="identity10",
        training_pin="a" * 40,
        packed_path="packed",
        packed_sha256="b" * 64,
        exposure_path="exposure",
        exposure_sha256="d" * 64,
        runtime_bytes=binding.runtime_bytes,
        runtime_sha256=binding.runtime_sha256,
        source_receipt={},
        source_objects={},
    )
    outcome = Future()
    with journey_world(tmp_path / "controller", monkeypatch) as world:
        root = world.submit("pathb-cpu-root")
        world.settle()
        execution = ExecutionBoundary(world, root)
        monkeypatch.setattr(workflow, "native_submit", execution.submit)

        def run():
            try:
                with iris_ctx_scope(context):
                    workflow.workflow_worker(training=training, binding=binding)
                outcome.set_result(None)
            except BaseException as error:
                outcome.set_exception(error)

        thread = threading.Thread(target=run)
        thread.start()
        inbox = root_events.get(timeout=10)
        train = execution.get("train")
        assert train.submit_options["env_vars"]["JAX_PLATFORMS"] == "cuda,cpu"
        assert train.kwargs["checkpoint_hook_factory"].root_actor_name == "/journey/pathb-cpu-root/checkpoint-inbox-0"
        inbox.enqueue(sample_event(25).payload())
        execution.get("export-000025")
        execution.finish("export-000025", success=False)
        # Failed diagnostic does not end root or scientific training.
        assert world.job(train.ref).state == job_pb2.JOB_STATE_RUNNING
        assert not outcome.done() and not hosted_stopped.is_set()
        second = sample_event(50)
        inbox.enqueue(second.payload())
        export = execution.get("export-000050")
        assert export.submit_options["env_vars"]["JAX_PLATFORMS"] == "cuda,cpu"
        local_storage(second.hf_export + "/path-b-checkpoint-event.json").write_text(json.dumps(second.payload()))
        execution.finish("export-000050")
        trend = execution.get("trend-000050")
        assert export.submit_options["lifetime"] == 7200
        assert len(world.tasks(export.ref)) == 1
        # Natural success releases the one-node export task; sibling trend lives.
        assert world.job(export.ref).state == job_pb2.JOB_STATE_SUCCEEDED
        assert all(t.state == job_pb2.TASK_STATE_SUCCEEDED for t in world.tasks(export.ref))
        assert world.job(trend.ref).state == job_pb2.JOB_STATE_RUNNING
        execution.finish("train")
        assert world.job(train.ref).state == job_pb2.JOB_STATE_SUCCEEDED
        assert world.job(trend.ref).state == job_pb2.JOB_STATE_RUNNING
        assert not outcome.done() and not hosted_stopped.is_set()
        execution.finish("trend-000050")
        outcome.result(timeout=10)
        thread.join(timeout=10)
        assert hosted_stopped.is_set() and not thread.is_alive()
        # Root only completes after all owned sibling handles became terminal.
        world.succeed_all(root)
        world.settle()
        assert world.job(root).state == job_pb2.JOB_STATE_SUCCEEDED
        assert receipts[training["run_root"] + "/workflow/final.json"]["missing_checkpoint_events"] == []
        assert all(j.ref.wire_id.rsplit("/", 1)[0] == root.wire_id for j in execution.jobs.values())


def test_inbox_rejects_conflicting_or_unbound_events_and_deduplicates_acknowledgements():
    messages = Queue()
    event = sample_event()
    inbox = workflow.CheckpointInbox(messages, {"arm": "identity10", "packed_data_sha256": "b" * 64}, (25,))
    inbox.enqueue(event.payload())
    inbox.enqueue(event.payload())
    assert messages.qsize() == 1
    with pytest.raises(ValueError, match="Conflicting"):
        inbox.enqueue(replace(event, exposure={"different": True}).payload())
    with pytest.raises(ValueError, match="cadence"):
        inbox.enqueue(sample_event(26).payload())
    with pytest.raises(ValueError, match="binding"):
        inbox.enqueue(replace(event, arm="identity25").payload())


def test_observation_transport_failure_keeps_handle_owned_until_terminal(monkeypatch):
    messages = Queue()
    calls = iter([ConnectionError("diagnostic read failed"), SimpleNamespace(state=JobState.SUCCEEDED)])

    def wait(**unused):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    # Read observations may resume, but no scientific job is resubmitted or
    # cancelled. Terminal is emitted only after a real terminal observation.
    monkeypatch.setattr(workflow.time, "sleep", lambda delay: None)
    workflow.observe_terminal(SimpleNamespace(job_id="owned", wait=wait), "trend", messages)
    assert messages.get_nowait() == ("observation_error", ("owned", "trend", "ConnectionError"))
    assert messages.get_nowait() == ("terminal", ("owned", "trend", JobState.SUCCEEDED))
