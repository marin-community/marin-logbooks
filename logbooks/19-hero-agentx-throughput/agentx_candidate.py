# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Use native AgentX orchestration with two recorded serving overrides.

The manifest pins the unchanged base recipe/runtime and separately identifies
the effective prefix-cache and private-wheel overrides. Replay, preparation,
warmup, benchmark duration and all native validity/attempt fences stay native.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from diagnostic_server import use_candidate_wheel
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
    receipt = use_candidate_wheel(server, Path(overrides["candidate_receipt"]), manifest["vllm_release"])
    assert receipt["wheel_sha256"] == overrides["wheel_sha256"]
    assert receipt["source_commit"] == overrides["source_commit"]
    server.extra_cli_args.remove("--no-enable-prefix-caching")
    server.extra_cli_args.append("--enable-prefix-caching")
    (output / "runtime-overrides.json").write_text(
        json.dumps({"requested": overrides, "wheel": receipt}, indent=2) + "\n"
    )
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    native.create_server = create_candidate_server
    signal.signal(signal.SIGTERM, native.interrupt)
    signal.signal(signal.SIGINT, native.interrupt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    monitor = None
    if os.environ["IRIS_TASK_ID"].endswith("/0"):
        monitor = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("client_load.py")),
                "--output",
                str(args.output.parent / "client-load.jsonl"),
                "--duration",
                "10800",
            ]
        )
    try:
        native.run(manifest, args.output)
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)


if __name__ == "__main__":
    main()
