# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit one four-GPU batch node for native CUTLASS preparation and a component check."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, gpu_device
from iris.rpc import job_pb2
from rigging.timing import Duration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert manifest["resources"] == {"gpus": 4, "cpu": 8, "memory": "256GB", "disk": "200GB"}
    assert manifest["placement"]["hero_training_gpus"] == 0
    assert manifest["placement"]["nodes"]
    assert time.time() - manifest["placement"]["checked_unix"] < 600
    command = (
        'exec env PYTHONPATH="$IRIS_WORKDIR" "$IRIS_PYTHON" '
        '"$IRIS_WORKDIR/logbooks/19-hero-agentx-throughput/cutlass_prepare.py" '
        '--manifest "$IRIS_WORKDIR/cutlass-prepare-manifest.json"'
    )
    entrypoint = Entrypoint(
        command=["bash", "-c", command],
        workdir_files={"cutlass-prepare-manifest.json": args.manifest.read_bytes()},
    )
    with connect_controller(cluster_name="cw-us-east-08a") as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials, workspace=Path.cwd()) as client:
            job = client.submit(
                entrypoint=entrypoint,
                name=args.name,
                resources=ResourceSpec(cpu=8, memory="256GB", disk="200GB", device=gpu_device("GB200", 4)),
                environment=EnvironmentSpec(
                    sync_packages=["marin-core"],
                    extras=["cpu"],
                    env_vars={"CUDA_VISIBLE_DEVICES": "", "FLASHINFER_CUDA_ARCH_LIST": "10.0a"},
                ),
                constraints=[],
                replicas=1,
                priority_band=job_pb2.PRIORITY_BAND_BATCH,
                scheduling_timeout=Duration.from_seconds(1800),
                timeout=Duration.from_seconds(10800),
                max_retries_failure=0,
                max_task_failures=0,
                max_retries_preemption=3,
                existing_job_policy=job_pb2.EXISTING_JOB_POLICY_KEEP,
                task_image=manifest["task_image"],
            )
            record = {
                "job_id": str(job.job_id),
                "source_commit": commit,
                "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "priority": "batch",
                "allocated_gpus": 4,
                "command": command,
                "submitted_unix": time.time(),
            }
            with args.output.open("x") as stream:
                json.dump(record, stream, indent=2)
                stream.write("\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
