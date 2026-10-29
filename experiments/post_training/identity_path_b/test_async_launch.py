# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import cast

import pytest
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.client import ClusterClient
from iris.cluster.types import JobName, TaskAttempt
from iris.rpc import job_pb2
from marin.evaluation.evalchemy.client import CONFIG_ENV_KEY, build_model_args
from marin.inference.types import OpenAIEndpoint, RunningModel

from experiments.post_training.identity_path_b import async_launch
from experiments.post_training.identity_path_b.async_launch import (
    CheckpointHookFactory,
    ExportBinding,
    TrendInputs,
)
from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent
from experiments.post_training.identity_path_b.trend import gsm_config, gsm_submission


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


def test_background_export_uses_native_parent_context_and_never_waits_for_child(monkeypatch, tmp_path):
    transport = SubmissionTransport()
    context = native_context(monkeypatch, transport)

    def local_storage(path):
        result = tmp_path / path.removeprefix("s3://marin-rno2a/tmp/ttl=14d/")
        result.parent.mkdir(parents=True, exist_ok=True)
        return result

    monkeypatch.setattr(async_launch, "StoragePath", local_storage)
    receipt = json.dumps({"xla_precision": "no_excess_precision"}).encode()
    binding = ExportBinding("fixture", receipt, "c" * 64, {}, {}, TrendInputs("identity", "controls"))
    with iris_ctx_scope(context):
        hook = CheckpointHookFactory("s3://marin-rno2a/tmp/ttl=14d/path-b/fixture", binding)()
        name = hook.publish_committed(event())
        hook.finish_submissions()
    result = json.loads(
        local_storage(
            f"s3://marin-rno2a/tmp/ttl=14d/path-b/fixture/checkpoint-events/{name}-submission.json"
        ).read_bytes()
    )
    assert result["submitted"]
    assert result["jobs"]["export"] == "/karan/pathb-fixture/export-000025"
    request = transport.submissions[0]
    assert request["replicas"] == 4
    assert request["ports"] == ["jax"]
    assert request["resources"].device.gpu.count == 8
    assert request["max_retries_failure"] == request["max_retries_preemption"] == 0
    assert request["priority_band"] == job_pb2.PRIORITY_BAND_INTERACTIVE
    assert request["timeout"].to_seconds() == 43200
    assert request["environment"].env_vars["XLA_FLAGS"] == "--xla_allow_excess_precision=false"
    assert request["environment"].env_vars["UNCHANGED_PARENT_ENV"] == "fixture"
    assert list(request["environment"].setup_scripts) == ["reviewed-parent-setup"]
    # OPD ambient children do not send another federation cluster constraint.
    assert not request["constraints"]
    pending = json.loads(local_storage(event().eval_scratch + "/trend-pending.json").read_bytes())
    assert pending["event"]["exposure"]["identity_final_text_tokens"] == 150000


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


def test_partial_or_mismatched_export_never_admits_an_eval_child(monkeypatch, tmp_path):
    transport = SubmissionTransport()
    context = native_context(monkeypatch, transport)

    def local_storage(path):
        result = tmp_path / path.removeprefix("s3://marin-rno2a/tmp/ttl=14d/")
        result.parent.mkdir(parents=True, exist_ok=True)
        return result

    monkeypatch.setattr(async_launch, "StoragePath", local_storage)
    # The full cleared worker is independently tested. This test exercises
    # only the child-admission boundary after the worker returns.
    monkeypatch.setattr(async_launch, "export_worker", lambda **kwargs: None)
    monkeypatch.setattr(async_launch.jax, "process_index", lambda: 0)
    binding = ExportBinding("fixture", b"{}", "c" * 64, {}, {}, TrendInputs("identity", "controls"))
    local_storage(event().hf_export + "/config.json").write_text("{}")
    marker = local_storage(event().hf_export + "/path-b-checkpoint-event.json")
    with iris_ctx_scope(context):
        with pytest.raises(FileNotFoundError):
            async_launch.export_then_submit_trend(event(), binding)
        marker.write_text(json.dumps({**event().payload(), "completed_updates": 99}))
        with pytest.raises(ValueError, match="completion event"):
            async_launch.export_then_submit_trend(event(), binding)
        assert not transport.submissions
        marker.write_text(json.dumps(event().payload()))
        async_launch.export_then_submit_trend(event(), binding)
    assert str(transport.submissions[0]["job_id"]) == "/karan/pathb-fixture/trend-000025"
    assert transport.submissions[0]["timeout"].to_seconds() == 86400

    def bad_export(**kwargs):
        raise ValueError("fixture failed storage audit")

    marker.unlink()
    monkeypatch.setattr(async_launch, "export_worker", bad_export)
    with iris_ctx_scope(context), pytest.raises(ValueError, match="storage audit"):
        async_launch.export_then_submit_trend(event(), binding)
    failure = json.loads(local_storage(event().eval_scratch + "/export-failed-task-0.json").read_bytes())
    assert failure["status"] == "export_failed" and failure["event"] == event().payload()
    assert len(transport.submissions) == 1
