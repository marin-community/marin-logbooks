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

from diagnostic_server import use_candidate_wheel
from iris.cluster.client.job_info import get_job_info
from marin.inference.iris_vllm import IrisVllmLaunch
from marin.inference.vllm_server import VllmEnvironment

from experiments.evaluation.agentx import run as native
from experiments.evaluation.agentx.recipe import PointManifest

BASE_CREATE_SERVER = native.create_server


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
    for key, flag in (("max_num_batched_tokens", "--max-num-batched-tokens"), ("master_port", "--master-port")):
        if key in overrides:
            server.extra_cli_args[server.extra_cli_args.index(flag) + 1] = str(overrides[key])
    if "master_port" in overrides and "--data-parallel-rpc-port" in server.extra_cli_args:
        server.extra_cli_args[server.extra_cli_args.index("--data-parallel-rpc-port") + 1] = str(
            overrides["master_port"] + 1
        )
    (output / "runtime-overrides.json").write_text(
        json.dumps({"requested": overrides, "wheel": receipt}, indent=2) + "\n"
    )
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    native.create_server = create_candidate_server
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
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
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)


if __name__ == "__main__":
    main()
