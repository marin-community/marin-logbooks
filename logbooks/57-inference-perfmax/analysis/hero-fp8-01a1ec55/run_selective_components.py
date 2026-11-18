# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Allocate one GB200 for the declared selective-FP8 component diagnostics."""

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, gpu_device
from iris.rpc import job_pb2
from marin.inference.vllm_server import IsolatedCudaVllm, VllmType
from rigging.timing import Duration

from experiments.evaluation.knotmeter.allocation import allocation_cost, track_allocations
from experiments.evaluation.knotmeter.artifacts import copy_file, retain_artifacts, write_record


def component_worker(payload: dict) -> None:
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    bootstrap = Path(payload["bootstrap"]).resolve()
    if hashlib.sha256(bootstrap.read_bytes()).hexdigest() != payload["bootstrap_sha256"]:
        raise RuntimeError("The component bootstrap differs from the published input")
    archive = output / "source.tar.gz"
    copy_file(payload["archive_uri"], str(archive))
    if hashlib.sha256(archive.read_bytes()).hexdigest() != payload["archive_sha256"]:
        raise RuntimeError("The component source archive checksum differs")
    root = output / "source"
    root.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    payload |= {"source_root": str(root), "output": str(output / "components.json")}
    config = output / "component-input.json"
    config.write_text(json.dumps(payload))
    launcher = IsolatedCudaVllm(
        source=VllmType.MARIN_FORK,
        extra_requirements=("pytest==9.0.3", "tblib==3.2.2"),
    )
    environment = os.environ | launcher.env()
    command = launcher.command()
    subprocess.run([*command, "--version"], env=environment, check=True)
    # Keep the native isolated environment and CUDA bootstrap; replace its CLI payload.
    boundary = command.index("python")
    command = [*command[: boundary + 3], "python", str(bootstrap), str(config)]
    try:
        subprocess.run(command, cwd=output, env=environment, check=True)
    finally:
        uri, digest = retain_artifacts(output, output.parent / "component.tar.zst", payload["uri"], name="native")
        write_record(payload["uri"] + "/receipt.json", {"archive_uri": uri, "archive_sha256": digest})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text())
    with connect_controller(cluster_name="cw-us-east-08a") as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials, workspace=Path.cwd()) as client:
            job = client.submit(
                entrypoint=Entrypoint.from_callable(component_worker, payload),
                name=payload["job_name"],
                resources=ResourceSpec(cpu=4, memory="32GB", disk="128GB", device=gpu_device("GB200", 1)),
                environment=EnvironmentSpec(sync_packages=["marin-core"], extras=["cpu"]),
                priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
                scheduling_timeout=Duration.from_seconds(1800),
                timeout=Duration.from_seconds(3600),
                max_retries_failure=0,
                max_retries_preemption=1,
                task_image=payload["task_image"],
            )
            plan = {"gpu_job_id": str(job.job_id), "cluster": "cw-us-east-08a", "trial_uri": payload["uri"]}
            args.output.write_text(json.dumps({"plan": plan, "payload": payload}, indent=2) + "\n")
            write_record(payload["uri"] + "/input.json", {"plan": plan, "payload": payload})
            print(json.dumps(plan), flush=True)
            with track_allocations(client, plan):
                job.wait(timeout=5400, raise_on_failure=False, stream_logs=True)
            attempts = []
            for summary in client.list_tasks(job.job_id):
                task = client.task_status(summary.task_id)
                for attempt in task.attempts:
                    finish = attempt.finished_at.epoch_ms() / 1000 if attempt.finished_at else None
                    cost, allocation = allocation_cost(plan, attempt.attempt_uid, attempt.pod_uid, finish, 1)
                    attempts.append(
                        {
                            "uid": attempt.attempt_uid,
                            "state": attempt.state.value,
                            "gpu_seconds": cost,
                            "allocation": allocation,
                            "error": attempt.error_message or attempt.terminal_reason,
                        }
                    )
            result = {
                "plan": plan,
                "state": client.job_status(job.job_id).state.value,
                "attempts": attempts,
                "observed_unix": time.time(),
            }
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            write_record(payload["uri"] + "/execution.json", result)
            print(json.dumps(result), flush=True)
            raise SystemExit(0 if result["state"] == "succeeded" else 1)


if __name__ == "__main__":
    main()
