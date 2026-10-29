# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Use native AgentX orchestration with recorded serving overrides.

The manifest pins the unchanged base recipe/runtime and separately identifies
the effective prefix-cache and private-wheel overrides. Replay, preparation,
warmup, benchmark duration and all native validity/attempt fences stay native.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from diagnostic_server import stage_candidate_wheel, use_candidate_wheel, use_numa_binding
from iris.cluster.client.job_info import get_job_info
from marin.inference.iris_vllm import IrisVllmLaunch
from marin.inference.vllm_server import VllmEnvironment
from router_process import upstream_router

from experiments.evaluation.agentx import run as native
from experiments.evaluation.agentx.recipe import PointManifest

BASE_CREATE_SERVER = native.create_server
BASE_WORKLOAD_ENVIRONMENT = native.workload_environment
BASE_BENCHMARK_COMMAND = native.benchmark_command
BASE_RUN_LOGGED = native.run_logged
SESSION_HEADER_ENV = "AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID"


def affinity_environment(bundle: Path, series: dict) -> dict[str, str]:
    return BASE_WORKLOAD_ENVIRONMENT(bundle, series) | {SESSION_HEADER_ENV: "1"}


def create_candidate_server(
    manifest: PointManifest, output: Path, launch: IrisVllmLaunch | None = None
) -> VllmEnvironment:
    overrides = manifest["runtime_overrides"]
    assert overrides["enable_prefix_caching"] is True
    server = BASE_CREATE_SERVER(manifest, output, launch)
    receipt = None
    if "candidate_receipt" in overrides:
        receipt = use_candidate_wheel(server, Path(overrides["candidate_receipt"]), manifest["vllm_release"])
        assert receipt["wheel_sha256"] == overrides["wheel_sha256"]
        assert receipt["source_commit"] == overrides["source_commit"]
    server.extra_cli_args.remove("--no-enable-prefix-caching")
    server.extra_cli_args.append("--enable-prefix-caching")
    for key, flag in (
        ("max_num_batched_tokens", "--max-num-batched-tokens"),
        ("master_port", "--master-port"),
        ("gpu_memory_utilization", "--gpu-memory-utilization"),
    ):
        if key in overrides:
            server.extra_cli_args[server.extra_cli_args.index(flag) + 1] = str(overrides[key])
    if "master_port" in overrides and "--data-parallel-rpc-port" in server.extra_cli_args:
        server.extra_cli_args[server.extra_cli_args.index("--data-parallel-rpc-port") + 1] = str(
            overrides["master_port"] + 1
        )
    if overrides.get("numa_bind"):
        assert overrides["numa_bind"] is True
        use_numa_binding(server, manifest["numactl_input"])
    (output / "runtime-overrides.json").write_text(
        json.dumps(
            {
                "requested": overrides,
                "wheel": receipt,
                "client_environment": {SESSION_HEADER_ENV: "1"} if overrides.get("session_affinity") else {},
            },
            indent=2,
        )
        + "\n"
    )
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if "series_filename" in manifest:
        path = Path(__file__).with_name(manifest["series_filename"])
        native.series_path = lambda profile: path
    if "candidate_wheel_input" in manifest:
        receipt = stage_candidate_wheel(manifest["candidate_wheel_input"])
        assert str(receipt) == manifest["runtime_overrides"]["candidate_receipt"]
    native.create_server = create_candidate_server
    if manifest["runtime_overrides"].get("session_affinity"):
        native.workload_environment = affinity_environment
    if "router" in manifest:
        assert manifest["runtime_overrides"]["session_affinity"] is True

        def router_benchmark_command(series: dict, concurrency: int, url: str, output: Path) -> list[str]:
            command = BASE_BENCHMARK_COMMAND(series, concurrency, url, output)
            # Server metrics retain the direct backend URL and native polling.
            command[command.index("--url") + 1] = manifest["router"]["base_url"]
            return command

        def run_with_router(command: list[str], environment: dict, log: Path, *, timeout: int) -> None:
            if "agentx" not in command or "profile" not in command:
                BASE_RUN_LOGGED(command, environment, log, timeout=timeout)
                return
            with upstream_router(manifest["router"], log.parent):
                BASE_RUN_LOGGED(command, environment, log, timeout=timeout)

        native.benchmark_command = router_benchmark_command
        native.run_logged = run_with_router
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    observer = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("server_cpu.py")),
            "--output",
            str(output / "server-cpu.jsonl"),
            "--samples",
            "3600",
        ]
    )
    monitor = None
    job_info = get_job_info()
    assert job_info is not None
    if job_info.task_index == 0:
        monitor = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("client_load.py")),
                "--output",
                str(output / "client-load.jsonl"),
                "--duration",
                "10800",
            ]
        )
    try:
        native.run(manifest)
    finally:
        observer.terminate()
        observer.wait(timeout=10)
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)


if __name__ == "__main__":
    main()
