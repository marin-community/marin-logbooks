# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit one bounded native diagnostic server at batch priority.

The native Iris allocation, whole-attempt retries, environment and output
capture are unchanged. Placement is checked again by the serving entry point.
Run the fixed native controls after readiness, then stop and capture output.
"""

import argparse
import hashlib
import json
from pathlib import Path

from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import CoschedulingConfig, Entrypoint, EnvironmentSpec, ResourceSpec, gpu_device
from iris.rpc import job_pb2
from rigging.timing import Duration

from experiments.evaluation.agentx.recipe import load_series, series_path
from experiments.evaluation.agentx.runtime import runtime_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--moe-backend", choices=("triton", "auto"), default="triton")
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--pipeline-stages", type=int, choices=(1, 4), default=1)
    parser.add_argument("--numa-bind", action="store_true")
    parser.add_argument("--all2all-backend", choices=("allgather_reducescatter", "flashinfer_nvlink_one_sided"))
    parser.add_argument("--serve-duration", type=int, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    path = series_path(manifest["profile"])
    series = load_series(path)
    assert manifest["profile"] == "gb200"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["series_sha256"]
    assert json.dumps(manifest["vllm_release"], sort_keys=True) == json.dumps(runtime_metadata("gb200"), sort_keys=True)
    assert args.budget > 0 and args.serve_duration > 0
    hardware = series["profiles"]["gb200"]
    command = (
        'exec env PYTHONPATH="$IRIS_WORKDIR" "$IRIS_PYTHON" '
        '"$IRIS_WORKDIR/logbooks/19-hero-agentx-throughput/diagnostic_server.py" '
        '--manifest "$IRIS_WORKDIR/diagnostic-manifest.json" --output "$IRIS_OUTPUT_DIR/diagnostic" '
        f"--prefix-cache --max-batched-tokens {args.budget} --serve-duration {args.serve_duration} "
        f"--moe-backend {args.moe_backend} --run-controls"
    )
    command += f" --pipeline-stages {args.pipeline_stages}"
    if args.numa_bind:
        command += " --numa-bind"
    if args.master_port is not None:
        command += f" --master-port {args.master_port}"
    if args.all2all_backend is not None:
        command += f" --all2all-backend {args.all2all_backend}"
    entrypoint = Entrypoint(
        command=["bash", "-c", command],
        workdir_files={"diagnostic-manifest.json": args.manifest.read_bytes()},
    )
    with connect_controller(cluster_name="cw-us-east-08a") as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials, workspace=Path.cwd()) as client:
            job = client.submit(
                entrypoint=entrypoint,
                name=args.name,
                resources=ResourceSpec(
                    cpu=hardware["cpu"],
                    memory=hardware["memory"],
                    disk=hardware["disk"],
                    device=gpu_device(hardware["gpu"], hardware["gpus"]),
                ),
                environment=EnvironmentSpec(sync_packages=["marin-core"], extras=["cpu"]),
                constraints=[],
                replicas=hardware["hosts"],
                coscheduling=CoschedulingConfig(group_by="nvlink.domain"),
                priority_band=job_pb2.PRIORITY_BAND_BATCH,
                scheduling_timeout=Duration.from_seconds(series["scheduling_timeout"]),
                timeout=Duration.from_seconds(series["ready_timeout"] + args.serve_duration + 600),
                max_retries_failure=0,
                max_task_failures=0,
                max_retries_preemption=series["max_retries_preemption"],
                existing_job_policy=job_pb2.EXISTING_JOB_POLICY_KEEP,
                task_image=series["task_image"],
            )
            record = {"job_id": str(job.job_id), "priority": "batch", "command": command, "manifest": manifest}
            args.output.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
