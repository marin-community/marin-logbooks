# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific options for the Snowball e2es.

The ``marin_gpu_client`` fixture these tests use lives in the shared
``tests/cluster/conftest.py``; this module only adds the attention-backend option.
"""

import contextlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest
from iris.cluster.client.job_info import get_job_info
from marin.inference.iris_vllm import (
    iris_vllm_followers,
    iris_vllm_launch,
    notify_iris_vllm_stopped,
    wait_for_iris_vllm_shutdown,
)
from marin.testing.inference.goldens import ModelFixture, read_fixture
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.knotmeter.artifacts import retain_artifacts, write_record
from experiments.evaluation.knotmeter.config import Profile
from experiments.evaluation.knotmeter.run import create_server

VLLM_ATTENTION_BACKENDS = ("FLASH_ATTN", "TRITON_ATTN")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--vllm-attention-backend",
        choices=VLLM_ATTENTION_BACKENDS,
        default="FLASH_ATTN",
        help="Attention backend for the Snowball vLLM e2e.",
    )
    parser.addoption("--correctness-plan", type=Path, help="Pinned deployment plan supplied by the Iris launcher")
    parser.addoption("--endpoint-url", help="Check an existing endpoint without deploying")
    parser.addoption("--correctness-fixture", help="JSON fixture path for an existing endpoint")
    parser.addoption("--correctness-model", help="Served model name for an existing endpoint")
    parser.addoption("--runtime-source", help="Expected vLLM source SHA for an existing endpoint")
    parser.addoption("--correctness-output", type=Path, help="Retained JSON report path")


@pytest.fixture
def vllm_attention_backend(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--vllm-attention-backend"))


@dataclass(frozen=True)
class CorrectnessEndpoint:
    url: str
    model: str
    fixture: ModelFixture
    runtime_source: str
    output: Path


@pytest.fixture
def correctness_endpoint(request: pytest.FixtureRequest, tmp_path):
    """Provide an endpoint and clean up any serving gang deployed by this fixture."""
    plan_path = request.config.getoption("--correctness-plan")
    explicit_output = request.config.getoption("--correctness-output")
    if plan_path is None:
        url = request.config.getoption("--endpoint-url")
        if url is None:
            pytest.skip("Supply an existing endpoint or submit a pinned correctness plan through Iris")
        yield CorrectnessEndpoint(
            url=url,
            model=request.config.getoption("--correctness-model"),
            fixture=read_fixture(request.config.getoption("--correctness-fixture")),
            runtime_source=request.config.getoption("--runtime-source"),
            output=explicit_output or tmp_path / "correctness.json",
        )
        return
    plan = json.loads(Path(plan_path).read_bytes())
    profile = Profile.model_validate(plan["series"]["profile"])
    if profile.serving.environment.get("HERO_SCHEDULER_TRACE") == "1":
        trace = Path(__file__).resolve().parents[3] / "logbooks/57-inference-perfmax/analysis/hero_trace_hook"
        previous = profile.serving.environment.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
        environment = profile.serving.environment | {"PYTHONPATH": os.pathsep.join(filter(None, (str(trace), previous)))}
        profile = profile.model_copy(update={"serving": profile.serving.model_copy(update={"environment": environment})})
        plan["series"]["profile"] = profile.model_dump(mode="json")
    fixture_names = [check.fixture for check in profile.checks if check.fixture]
    if len(fixture_names) != 1:
        raise ValueError("A deployment gate requires exactly one shared model fixture")
    fixture = read_fixture(fixture_names[0])
    info = get_job_info()
    if info is None:
        raise RuntimeError("Submit this deployment plan through the inference_correctness launcher")
    output = Path(os.environ["IRIS_OUTPUT_DIR"]) / "correctness"
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "series": plan["series"],
        "vllm_release": plan["vllm_release"],
        "source": plan["source"],
        "placement": plan["placement"],
        "workload": {},
        "profile": "h100" if profile.gpus.type == "H100" else "gb200",
        "concurrency": profile.session_concurrency,
        "recipe_fingerprint": "correctness-only",
        "diagnostic": plan.get("diagnostic", False),
    }
    topology = (
        iris_vllm_launch(
            pipeline_parallel_size=1,
            tensor_parallel_size=profile.serving.tensor_parallel_size,
            data_parallel_size=profile.gpus.count // profile.serving.tensor_parallel_size,
            data_parallel_size_local=profile.gpus.per_host // profile.serving.tensor_parallel_size,
        )
        if profile.gpus.count > profile.gpus.per_host
        else contextlib.nullcontext(None)
    )
    log_directory = None
    try:
        with topology as launch:
            with create_server(manifest, output, launch) as server:
                assert server.vllm_server is not None
                log_directory = Path(server.vllm_server.log_dir)
                write_record(str(output / "deployment.json"), {"plan": plan, "command": server.command})
                if launch is not None and not launch.is_leader:
                    wait_for_iris_vllm_shutdown(launch, server.check_alive)
                else:
                    server.wait_until_ready()
                    with iris_vllm_followers(launch) if launch else contextlib.nullcontext():
                        yield CorrectnessEndpoint(
                            server.worker_server_url,
                            profile.model.repo,
                            fixture,
                            plan["vllm_release"]["source_commit"],
                            explicit_output or output / "report.json",
                        )
            if launch is not None and not launch.is_leader:
                notify_iris_vllm_stopped(launch)
                pytest.skip("Follower serving process completed; the leader owns numerical assertions")
    finally:
        if log_directory is not None:
            shutil.copytree(log_directory, output / "server-logs")
        prefix = prefix_join(plan["trial_uri"], f"attempts/{os.environ['IRIS_ATTEMPT_UID']}/hosts/{info.task_index}")
        uri, digest = retain_artifacts(output, output.parent / "correctness.tar.zst", prefix, name="native")
        write_record(
            prefix_join(prefix, "receipt.json"),
            {
                "archive_uri": uri,
                "archive_sha256": digest,
                "source": plan["source"],
                "task_index": info.task_index,
                "attempt_uid": os.environ["IRIS_ATTEMPT_UID"],
            },
        )
