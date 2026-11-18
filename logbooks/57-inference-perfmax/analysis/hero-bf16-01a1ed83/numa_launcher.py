# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Use the native Knotmeter lifecycle after staging a pinned numactl binary."""

import hashlib
import json
import os
from pathlib import Path

from marin.inference.iris_vllm import IrisVllmLaunch
from marin.inference.vllm_server import VllmEnvironment, VllmLauncherWithEnvironment

from experiments.evaluation.knotmeter import cli, coordinator
from experiments.evaluation.knotmeter import run as native
from experiments.evaluation.knotmeter.recipe import PointManifest, TrialPlan

from .numa_probe import PACKAGE, command_result, prepare_numactl

BASE_CREATE_SERVER = native.create_server


def create_numa_server(manifest: PointManifest, output: Path, launch: IrisVllmLaunch | None) -> VllmEnvironment:
    args = manifest["series"]["profile"]["serving"]["extra_args"]
    if "--numa-bind" not in args:
        raise ValueError("The resolved profile must declare native NUMA binding")
    binary = prepare_numactl()
    setup = {
        "package": PACKAGE,
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "runtime_source": manifest["vllm_release"]["source_commit"],
        "setup_source_sha256": (
            hashlib.sha256(
                (
                    Path(native.__file__).resolve().parents[3]
                    / "logbooks/57-inference-perfmax/analysis/hero-bf16-01a1ed83/numa_launcher.py"
                ).read_bytes()
            ).hexdigest()
        ),
        "scope": "Extract a private dependency and extend this vLLM subprocess PATH. Runtime code is unchanged.",
        "policy_probes": [
            command_result([str(binary), f"--cpunodebind={node}", f"--membind={node}", str(binary), "--show"])
            for node in (0, 1)
        ],
    }
    (output / "numactl-setup.json").write_text(json.dumps(setup, indent=2) + "\n")
    if any(probe["returncode"] for probe in setup["policy_probes"]):
        raise RuntimeError("Native CPU-and-memory binding failed before server startup")
    server = BASE_CREATE_SERVER(manifest, output, launch)
    server.launcher = VllmLauncherWithEnvironment(
        server.launcher, {"PATH": str(binary.parent) + os.pathsep + os.environ["PATH"]}
    )
    return server


def run_numa_point(plan: TrialPlan) -> None:
    native.create_server = create_numa_server
    native.run(plan)


def run_numa_coordinator(plan: TrialPlan) -> None:
    coordinator.run = run_numa_point
    coordinator.run_coordinator(plan)


if __name__ == "__main__":
    cli.run_coordinator = run_numa_coordinator
    cli.main()
