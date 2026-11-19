# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one CPU packaging check or one native GPU padding diagnostic."""

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, gpu_device
from iris.rpc import job_pb2
from marin.inference.vllm_server import IsolatedCudaVllm, VllmType
from rigging.timing import Duration

from experiments.evaluation.knotmeter.allocation import allocation_cost, track_allocations
from experiments.evaluation.knotmeter.artifacts import copy_file, retain_artifacts, write_record
from experiments.evaluation.knotmeter.runtime import materialize_release


def component_worker(payload: dict) -> None:
    if payload.get("mode") == "compare":
        for label, variant in payload["variants"].items():
            component_worker(
                payload | variant | {"mode": "component", "label": label, "uri": payload["uri"] + "/" + label}
            )
        return
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    if payload.get("label"):
        output /= payload["label"]
        output.mkdir()
    bootstrap = Path(payload["bootstrap"]).resolve()
    if hashlib.sha256(bootstrap.read_bytes()).hexdigest() != payload["bootstrap_sha256"]:
        raise RuntimeError("The component bootstrap differs from the published input")
    work = Path(tempfile.mkdtemp(prefix="hero-fp8-source-"))
    archive = work / "source.tar.gz"
    copy_file(payload["archive_uri"], str(archive))
    if hashlib.sha256(archive.read_bytes()).hexdigest() != payload["archive_sha256"]:
        raise RuntimeError("The component source archive checksum differs")
    root = work / "source"
    root.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    payload |= {"source_root": str(root), "output": str(output / "components.json")}
    requirements = ["pytest==9.0.3", "tblib==3.2.2"]
    if payload.get("mode") == "build":
        wheel = work / "base.whl"
        with urllib.request.urlopen(payload["base_wheel_url"], timeout=120) as reader, wheel.open("wb") as writer:
            while chunk := reader.read(8 * 1024**2):
                writer.write(chunk)
        with wheel.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != payload["base_wheel_sha256"]:
                raise RuntimeError("The precompiled wheel differs from the pinned base")
        payload["base_wheel"] = str(wheel)
        requirements += [
            "build==1.6.1",
            "cmake==4.4.4",
            "ninja==1.13.2",
            "packaging==26.3",
            "setuptools==80.10.2",
            "setuptools-scm==10.3.4",
            "setuptools-rust==1.13.0",
            "wheel==0.48.0",
        ]
    config = output / "component-input.json"
    config.write_text(json.dumps(payload))
    launcher = IsolatedCudaVllm(
        source=VllmType.MARIN_FORK,
        extra_requirements=tuple(requirements),
        **(
            {"release": materialize_release(payload["candidate_manifest"], output)}
            if payload.get("candidate_manifest")
            else {}
        ),
    )
    environment = os.environ | launcher.env()
    command = launcher.command()
    if payload.get("mode") != "build":
        subprocess.run([*command, "--version"], env=environment, check=True)
    # Keep the native isolated environment and CUDA bootstrap; replace its CLI payload.
    boundary = command.index("python")
    command = [*command[: boundary + 3], "python", str(bootstrap), str(config)]
    try:
        with (output / "native.log").open("w") as log:
            subprocess.run(command, cwd=output, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        if payload.get("mode") == "build":
            built = json.loads(Path(payload["output"]).read_text())
            copy_file(built["wheel_path"], payload["uri"] + "/" + Path(built["wheel_path"]).name)
            manifest = {
                "release_tag": payload["release_tag"],
                "source_commit": payload["source_commit"],
                "version": built["version"],
                "torch_backend": "cu132",
                "torch_version": "2.14.1+cu132",
                "wheels": [
                    {
                        "architecture": "aarch64",
                        "sm_targets": ["10.0"],
                        "url": payload["uri"] + "/" + Path(built["wheel_path"]).name,
                        "sha256": built["wheel_sha256"],
                    }
                ],
            }
            write_record(payload["uri"] + "/manifest.json", manifest)
            write_record(payload["uri"] + "/build.json", built)
    finally:
        uri, digest = retain_artifacts(output, output.parent / "component.tar.zst", payload["uri"], name="native")
        write_record(payload["uri"] + "/receipt.json", {"archive_uri": uri, "archive_sha256": digest})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text())
    bootstrap = Path(payload["bootstrap"])
    if hashlib.sha256(bootstrap.read_bytes()).hexdigest() != payload["bootstrap_sha256"]:
        raise ValueError("The local bootstrap differs from the frozen input; no job submitted")
    gpus = 0 if payload.get("mode") == "build" else 1
    with connect_controller(cluster_name="cw-us-east-08a") as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials, workspace=Path.cwd()) as client:
            job = client.submit(
                entrypoint=Entrypoint.from_callable(component_worker, payload),
                name=payload["job_name"],
                resources=ResourceSpec(
                    cpu=8 if not gpus else 4,
                    memory="32GB",
                    disk="128GB",
                    device=gpu_device("GB200", gpus) if gpus else None,
                ),
                environment=EnvironmentSpec(sync_packages=["marin-core"], extras=["cpu"]),
                priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
                scheduling_timeout=Duration.from_seconds(1800),
                timeout=Duration.from_seconds(1800),
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
                    cost, allocation = allocation_cost(plan, attempt.attempt_uid, attempt.pod_uid, finish, gpus)
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
