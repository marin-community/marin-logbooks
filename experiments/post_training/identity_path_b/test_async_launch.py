# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from queue import Queue
from types import SimpleNamespace
from typing import cast

import pytest
from iris.actor.resolver import FixedResolver
from iris.actor.server import ActorServer
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.client import ClusterClient
from iris.cluster.types import Entrypoint, JobName, TaskAttempt
from iris.rpc import job_pb2
from marin.evaluation.evalchemy.client import CONFIG_ENV_KEY, build_model_args
from marin.inference.types import OpenAIEndpoint, RunningModel

from experiments.post_training.identity_context_distill.lr0_probe import RESOURCES
from experiments.post_training.identity_path_b import async_launch
from experiments.post_training.identity_path_b.async_launch import (
    CheckpointHookFactory,
    ExportBinding,
    TrendInputs,
)
from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent
from experiments.post_training.identity_path_b.trend import gsm_config, gsm_submission
from experiments.post_training.identity_path_b.workflow import CheckpointInbox


class SubmissionTransport:
    """Public native submit transport; completion/status APIs deliberately absent."""

    def __init__(self):
        self.submissions = []

    def submit_job(self, **kwargs):
        self.submissions.append(kwargs)
        return kwargs["job_id"]


def event():
    root = "s3://marin-rno2a/tmp/ttl=14d/path-b/fixture"
    return CheckpointEvent(
        25,
        f"{root}/checkpoints/step-24",
        f"{root}/hf/updates-25",
        f"{root}/eval/updates-25",
        {"identity_final_text_tokens": 150000},
        "a" * 40,
        "b" * 64,
        "identity10",
    )


def native_context(monkeypatch, transport):
    monkeypatch.setenv("IRIS_TASK_ID", "/karan/pathb-fixture/0:0")
    monkeypatch.setenv("IRIS_CONTROLLER_URL", "http://assigned-fixture")
    monkeypatch.setenv("IRIS_CONTROLLER_ADDRESS", "http://assigned-fixture")
    monkeypatch.setenv("IRIS_JOB_ENV", json.dumps({"UNCHANGED_PARENT_ENV": "fixture"}))
    monkeypatch.setenv("IRIS_JOB_SETUP_SCRIPTS", json.dumps(["reviewed-parent-setup"]))
    return IrisContext(
        job_id=JobName.from_wire("/karan/pathb-fixture"),
        task_attempt=TaskAttempt.from_wire("/karan/pathb-fixture/0:0"),
        client=IrisClient(cast(ClusterClient, transport)),
    )


def test_checkpoint_callback_enqueues_over_native_rpc_without_waiting_for_diagnostics(monkeypatch, tmp_path):

    messages = Queue()
    inbox = CheckpointInbox(messages, {"arm": "identity10"}, (25,))
    server = ActorServer(host="127.0.0.1", port=0)
    actor_name = "/karan/root/checkpoint-inbox-0"
    server.register(actor_name, inbox)
    port = server.serve_background()
    resolver = FixedResolver({actor_name: f"http://127.0.0.1:{port}"})
    monkeypatch.setattr(async_launch, "regional_task_context", lambda: SimpleNamespace(resolver=resolver))

    def local_storage(path):
        result = tmp_path / path.removeprefix("s3://marin-rno2a/tmp/ttl=14d/")
        result.parent.mkdir(parents=True, exist_ok=True)
        return result

    monkeypatch.setattr(async_launch, "StoragePath", local_storage)
    try:
        hook = CheckpointHookFactory("s3://marin-rno2a/tmp/ttl=14d/path-b/fixture", actor_name)()
        name = hook.publish_committed(event())
        hook.finish_submissions()
        kind, received = messages.get_nowait()
        assert kind == "checkpoint" and received == event()
        assert messages.empty()
        result = json.loads(
            local_storage(
                f"s3://marin-rno2a/tmp/ttl=14d/path-b/fixture/checkpoint-events/{name}-submission.json"
            ).read_bytes()
        )
        assert result["submitted"] and result["jobs"]["cpu_root_enqueued"]
        # There are no export/eval job handles in the actor or GPU callback.
        assert "export" not in result["jobs"]
    finally:
        server.stop()


def test_gsm_native_payload_preserves_historical_decode_and_transport_with_explicit_job_policy(monkeypatch):
    transport = SubmissionTransport()
    context = native_context(monkeypatch, transport)
    model = RunningModel(OpenAIEndpoint("http://fixture/v1", event().hf_export), tokenizer=event().hf_export)
    config = gsm_config(event())
    with iris_ctx_scope(context):
        job = gsm_submission(model, config, event().eval_scratch + "/gsm8k_0shot")
    assert str(job.job_id) == "/karan/pathb-fixture/gsm8k-one-decode"
    request = transport.submissions[0]
    payload = json.loads(request["environment"].env_vars[CONFIG_ENV_KEY])
    assert len(payload["tasks"]) == 1
    assert payload["tasks"][0]["name"] == "gsm8k"
    assert payload["tasks"][0]["num_fewshot"] == 0
    assert payload["max_eval_instances"] is None
    assert payload["seed"] == 0
    assert payload["extra_gen_kwargs"] == {"temperature": "0"}
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["max_gen_toks"] == 8192 and payload["max_length"] == 32768
    assert payload["dataset_cache"].startswith(event().eval_scratch + "/")
    actual_args = build_model_args(payload, True, 32704)
    assert "max_retries=8" in actual_args
    assert "transport_retry_budget=5400.0" in actual_args
    assert "transport_attempt_timeout=1800.0" in actual_args
    assert request["max_retries_failure"] == request["max_retries_preemption"] == 0
    assert request["timeout"].to_seconds() == 43200
    assert request["priority_band"] == job_pb2.PRIORITY_BAND_INTERACTIVE


def test_export_returns_without_descendants_and_requires_matching_audited_marker(monkeypatch, tmp_path):
    def local_storage(path):
        result = tmp_path / path.removeprefix("s3://marin-rno2a/tmp/ttl=14d/")
        result.parent.mkdir(parents=True, exist_ok=True)
        return result

    monkeypatch.setattr(async_launch, "StoragePath", local_storage)
    monkeypatch.setattr(async_launch, "export_worker", lambda **kwargs: None)
    monkeypatch.setattr(async_launch.jax, "process_index", lambda: 0)
    binding = ExportBinding("fixture", b"{}", "c" * 64, {}, {}, TrendInputs("identity", "controls"))
    marker = local_storage(event().hf_export + "/path-b-checkpoint-event.json")
    with pytest.raises(FileNotFoundError):
        async_launch.export_checkpoint(event(), binding)
    marker.write_text(json.dumps({**event().payload(), "completed_updates": 99}))
    with pytest.raises(ValueError, match="completion event"):
        async_launch.export_checkpoint(event(), binding)
    marker.write_text(json.dumps(event().payload()))
    async_launch.export_checkpoint(event(), binding)
    assert json.loads(marker.read_bytes()) == event().payload()


def test_native_submit_retains_four_rank_mesh_in_assigned_root_context(monkeypatch):

    transport = SubmissionTransport()
    context = native_context(monkeypatch, transport)
    with iris_ctx_scope(context):
        job = async_launch.native_submit(
            Entrypoint.from_command("true"),
            name="export-000025",
            resources=RESOURCES,
            replicas=4,
            env_vars={"XLA_FLAGS": "--xla_allow_excess_precision=false"},
            lifetime=43200,
        )
    assert str(job.job_id) == "/karan/pathb-fixture/export-000025"
    request = transport.submissions[0]
    assert request["replicas"] == 4 and request["ports"] == ["jax"]
    assert request["resources"].device.gpu.count == 8
    assert request["max_retries_failure"] == request["max_retries_preemption"] == 0
    assert request["priority_band"] == job_pb2.PRIORITY_BAND_INTERACTIVE
    assert request["timeout"].to_seconds() == 43200
    assert request["environment"].env_vars["UNCHANGED_PARENT_ENV"] == "fixture"
    assert list(request["environment"].setup_scripts) == ["reviewed-parent-setup"]
    assert not request["constraints"]


def test_optional_receipt_and_enqueue_failures_remain_diagnostics_not_training_errors(monkeypatch, caplog):

    class FailedActor:
        def __init__(self, *args, **kwargs):
            pass

        def enqueue(self, payload):
            raise ConnectionError("root enqueue unavailable")

    def unavailable_storage(path):
        raise OSError("optional receipt storage unavailable")

    monkeypatch.setattr(async_launch, "ActorClient", FailedActor)
    monkeypatch.setattr(async_launch, "regional_task_context", lambda: SimpleNamespace(resolver=None))
    monkeypatch.setattr(async_launch, "StoragePath", unavailable_storage)
    hook = CheckpointHookFactory("s3://marin-rno2a/tmp/ttl=14d/path-b/fixture", "root-inbox")()
    hook.publish_committed(event())
    hook.finish_submissions()
    assert "Diagnostic receipt write failed" in caplog.text
    assert "no retry" in caplog.text
