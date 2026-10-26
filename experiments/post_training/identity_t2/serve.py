# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native one-task regional serve blueprint for the reviewed CD16-T2 capture.

`plan` writes an allowlisted, credential-free description. `worker` only runs
inside Iris. This module deliberately has no submit/retry command: the executor
must first bind review, clean pin, current orders, fresh namespaces and capacity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fray.iris_backend import convert_constraints, convert_resources
from fray.types import ResourceConfig, create_environment
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from marin.inference.config import IrisConfig, ServedModelConfig, VllmEngineConfig, VllmLauncherType
from marin.inference.iris import IrisServiceConfig, run_iris_service
from rigging.secrets import SecretBehavior
from rigging.timing import Duration

from vendor.skyrl.cloud.iris.runtime_environment import IRIS_TASK_IMAGE, RuntimeProfile, task_setup_script

MODEL = "s3://marin-rno2a/marin/models/marin-snowball-67b-a2b-v0.1-adc0078a"
TIMEOUT_SECONDS = 7200
JOB_NAME = "cd16t2-generation-r1"
ENDPOINT_NAME = "/serve/" + JOB_NAME
PROFILE_ENV = "/app/vendor/skyrl/.cd16t2-vllm-runtime"


def setup_script() -> str:
    # Keep the native Marin worker interpreter intact. Provision the EXISTING
    # frozen SkyRL profile in a separate env solely for its pinned vLLM binary.
    # No lock edits, dependency substitution, worker patch or model monkeypatch.
    return (
        "set -euo pipefail\n(\n"
        f"export IRIS_VENV={PROFILE_ENV}\n"
        + task_setup_script(RuntimeProfile.MEGATRON)
        + "\n)\nsource /app/vendor/skyrl/.iris-runtime-env\n"
    )


def service_config() -> IrisServiceConfig:
    resources = ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="512g", disk="512g")
    return IrisServiceConfig(
        model=ServedModelConfig(weights=MODEL, dtype="bfloat16", tensor_parallel_size=1, max_model_len=32768),
        engine=VllmEngineConfig(
            launcher=VllmLauncherType.PREINSTALLED,
            startup_timeout_seconds=1800,
            max_num_seqs=32,
            extra_args=(
                "--data-parallel-size",
                "8",
                "--enable-expert-parallel",
                "--gpu-memory-utilization",
                "0.75",
                "--generation-config",
                "vllm",
            ),
        ),
        iris=IrisConfig(
            worker_resources=resources,
            worker_environment=create_environment(extras=["gpu"]),
            cache_ttl_days=0,
            max_retries_failure=0,
            max_retries_preemption=0,
        ),
        endpoint_name=ENDPOINT_NAME,
        timeout_hours=TIMEOUT_SECONDS / 3600,
        controller_proxy_timeout_seconds=360,
    )


def submission(credentials: dict[str, str]):
    """Actual native arguments; never dump this dictionary (it holds credentials)."""
    resources = ResourceConfig.with_gpu(
        "H100",
        count=8,
        cpu=32,
        ram="512g",
        disk="512g",
        target_cluster="cw-rno2a",
    )
    secrets = {
        "CW_KEY_ID": credentials["CW_KEY_ID"],
        "CW_KEY_SECRET": credentials["CW_KEY_SECRET"],
        "AWS_ACCESS_KEY_ID": credentials["CW_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": credentials["CW_KEY_SECRET"],
    }
    if credentials.get("HF_TOKEN"):
        secrets["HF_TOKEN"] = credentials["HF_TOKEN"]
    return {
        "name": JOB_NAME,
        "entrypoint": Entrypoint(
            command=[
                "bash",
                "-c",
                'exec "$IRIS_PYTHON" -u -m experiments.post_training.identity_t2.serve worker',
            ]
        ),
        "task_image": IRIS_TASK_IMAGE,
        "resources": convert_resources(resources),
        "constraints": convert_constraints(resources),
        "replicas": 1,
        "ports": ["http"],
        "environment": EnvironmentSpec(
            extras=["gpu"],
            secret_behavior=SecretBehavior.ISOLATE,
            secrets=secrets,
            setup_scripts=[setup_script()],
            env_vars={"UV_FROZEN": "1", "OMP_NUM_THREADS": "1", "VLLM_TARGET_DEVICE": "cuda"},
        ),
        "timeout": Duration.from_seconds(TIMEOUT_SECONDS),
        "max_retries_failure": 0,
        "max_retries_preemption": 0,
        "priority_band": job_pb2.PRIORITY_BAND_PRODUCTION,
    }


def safe_plan() -> dict:
    return {
        "job": "/karan/" + JOB_NAME,
        "endpoint_name_NOT_capability": ENDPOINT_NAME,
        "weights": MODEL,
        "source_revision": "adc0078a19a4383bc998fcdb43867b3360baf5f3",
        "resources": {"replicas": 1, "H100_per_task": 8, "CPU_per_task": 32, "RAM": "512g", "disk": "512g"},
        "placement": "cw-rno2a",
        "priority": "production",
        "native_timeout_seconds": TIMEOUT_SECONDS,
        "application_TTL_seconds": TIMEOUT_SECONDS,
        "failure_retries": 0,
        "preemption_retries": 0,
        "native_timeout_includes_setup_load_compile_capture_and_teardown": True,
        "nominal_H100h_ceiling_NOT_actual": TIMEOUT_SECONDS * 8 / 3600,
        "runtime_image": IRIS_TASK_IMAGE,
        "locked_profile": "vendor/skyrl existing megatron+vllm; separate env; no dependency or lock edits",
        "vllm_build": "0.0.0.dev20260929+marin.39e62869693c.cu132",
        "launcher": "native PreinstalledVllm resolves the locked profile binary on PATH",
        "dtype": "bfloat16",
        "TP": 1,
        "DP": 8,
        "EP": True,
        "gpu_memory_utilization": 0.75,
        "max_model_len": 32768,
        "max_num_seqs": 32,
        "auto_tool_choice": False,
        "tool_parser": None,
        "reasoning_parser": None,
        "generation_config": "vllm",
        "request_stop_token_ids": [128001, 128009],
        "enforce_eager": False,
        "strict_tool_calling_environment_override": False,
        "setup_script_sha256": hashlib.sha256(setup_script().encode()).hexdigest(),
        "secret_behavior": "ISOLATE",
        "secret_names_only": ["CW_KEY_ID", "CW_KEY_SECRET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN"],
        "executor_release": (
            "Executor finally cancels this exact owned job on completion or failure; no warm hold for QC"
        ),
        "status": "PLAN ONLY; no submission, runtime version or live-five pass claimed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "worker"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.mode == "worker":
        run_iris_service(service_config())
    else:
        if args.out is None:
            raise ValueError("plan needs --out")
        with args.out.open("x") as file:
            json.dump(safe_plan(), file, indent=2)
            file.write("\n")


if __name__ == "__main__":
    main()
