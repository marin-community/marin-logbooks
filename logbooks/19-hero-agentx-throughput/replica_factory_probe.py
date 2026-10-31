# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise private API command construction with this actual native JobInfo."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from diagnostic_server import use_private_api_binding
from iris.cluster.client.job_info import get_job_info
from marin.inference.iris_vllm import _build_task_launch
from marin.inference.vllm_server import _vllm_serve_command

from experiments.evaluation.agentx.run import create_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    info = get_job_info()
    assert info is not None and info.task_index == 0
    assert str(info.job_id) == manifest["job_id"] and info.num_tasks == manifest["num_tasks"]
    assert os.environ["IRIS_ATTEMPT_UID"] == manifest["attempt_uid"]
    uuids = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).split()
    assert set(uuids) == set(manifest["gpu_uuids"]) and len(uuids) == 4
    devices = info.num_tasks * len(uuids)
    launch = _build_task_launch(
        info,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        data_parallel_size=devices,
        data_parallel_size_local=len(uuids),
        master_addr=info.advertise_host,
        coordinator_name=str(info.job_id).strip("/").replace("/", "-") + "-vllm-coordinator",
    )
    args.output.mkdir(exist_ok=False)
    server = create_server(manifest, args.output, launch)
    binding = use_private_api_binding(server, launch, manifest["diagnostic_request"], False)
    command = _vllm_serve_command(
        launcher=server.launcher,
        model_name_or_path=server.model_name_or_path,
        host=server.host,
        port=server.port,
        extra_cli_args=server.extra_cli_args,
    )
    assert command[command.index("--host") + 1] == launch.host_ip
    assert command[command.index("--data-parallel-size") + 1] == str(devices)
    assert command[command.index("--nnodes") + 1] == str(info.num_tasks)
    assert server.vllm_server is None and "--enforce-eager" not in command
    result = {
        "status": "actual_native_unentered_private_api_factory_passed",
        "job_id": str(info.job_id),
        "num_tasks": info.num_tasks,
        "task_index": info.task_index,
        "native_group_devices": devices,
        "api_bind": binding,
        "gpu_uuids": uuids,
        "server_started": False,
        "command": command,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "scope": "Actual native launch and command assembly only. No private-network, DP16/32 or model validation.",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
