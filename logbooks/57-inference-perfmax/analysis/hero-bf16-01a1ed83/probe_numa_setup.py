# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the serialized native NUMA factory on an ARM CPU task."""

import argparse
import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

import cloudpickle
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.constraints import preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2
from rigging.timing import Duration

from experiments.evaluation.knotmeter.artifacts import write_record

from . import numa_launcher


def run_probe(manifest: dict, setup_factory, output_uri: str) -> None:
    if platform.machine() != "aarch64":
        raise ValueError("This setup probe must use the native ARM task placement")
    output = Path(os.environ["IRIS_OUTPUT_DIR"]) / "setup-probe"
    output.mkdir(parents=True, exist_ok=True)
    server = setup_factory(manifest, output, None)
    setup = json.loads((output / "numactl-setup.json").read_text())
    source = (
        Path(numa_launcher.native.__file__).resolve().parents[3]
        / "logbooks/57-inference-perfmax/analysis/hero-bf16-01a1ed83/numa_launcher.py"
    )
    assert setup["setup_source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert "--numa-bind" in server.extra_cli_args
    assert server.launcher.environment["PATH"].split(os.pathsep)[0] == str(Path(setup["binary"]).parent)
    assert server.vllm_server is None
    report = {
        "status": "passed",
        "observed_utc": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "architecture": platform.machine(),
        "iris_task_id": os.environ["IRIS_TASK_ID"],
        "iris_attempt_uid": os.environ["IRIS_ATTEMPT_UID"],
        "iris_node_name": os.environ.get("IRIS_NODE_NAME"),
        "serialized_factory_dunder_file": setup_factory.__globals__.get("__file__"),
        "resolved_remote_source": str(source),
        "setup": setup,
        "launcher_command": server.launcher.command(),
        "serving_extra_args": server.extra_cli_args,
        "native_subprocess_path_prefix": server.launcher.environment["PATH"].split(os.pathsep)[0],
        "server_started": False,
        "requested_gpus": 0,
        "scope": (
            "Real serialized NUMA factory, dependency staging, policy probes and native launcher construction. "
            "The server context is never entered. No model tensor or CUDA execution."
        ),
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    write_record(output_uri, report)
    print(json.dumps({"status": report["status"], "output_uri": output_uri, "architecture": report["architecture"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    # Match the by-value __main__ boundary that exposed the devbox __file__ bug.
    cloudpickle.register_pickle_by_value(numa_launcher)
    with connect_controller(cluster_name=plan["controller"]) as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials, workspace=Path.cwd()) as client:
            job = client.submit(
                entrypoint=Entrypoint.from_callable(
                    run_probe, plan["manifest"], numa_launcher.create_numa_server, args.output_uri
                ),
                name="hero-bf16-numa-setup-" + datetime.now(UTC).strftime("%Y%m%dt%H%M%Sz"),
                resources=ResourceSpec(cpu=2, memory="32GB", disk="64GB"),
                environment=EnvironmentSpec(
                    sync_packages=["marin-core"], extras=["cpu"], env_vars={"MARIN_PREFIX": plan["marin_prefix"]}
                ),
                constraints=[preemptible_constraint(False)],
                priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
                scheduling_timeout=Duration.from_seconds(300),
                timeout=Duration.from_seconds(600),
                max_retries_failure=0,
                max_retries_preemption=0,
                task_image=plan["series"]["task_image"],
            )
            submission = {
                "job_id": str(job.job_id),
                "output_uri": args.output_uri,
                "task_image": plan["series"]["task_image"],
                "requested_gpus": 0,
                "source_factory_sha256": hashlib.sha256(Path(numa_launcher.__file__).read_bytes()).hexdigest(),
            }
            args.output.write_text(json.dumps(submission, indent=2) + "\n")
            print(json.dumps(submission), flush=True)
            job.wait(timeout=900, raise_on_failure=True, stream_logs=True)


if __name__ == "__main__":
    main()
