# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hold one native Hero server for controlled requests inside the dev allocation.

This uses the benchmark's existing launch and serving code. It runs no AgentX
window. The leader stops on its local ``stop`` file or the bounded serving time.
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from marin.external_dependencies import VllmGpuRelease, VllmGpuWheel
from marin.inference.iris_vllm import (
    IrisVllmLaunch,
    iris_vllm_followers,
    iris_vllm_launch,
    notify_iris_vllm_stopped,
    wait_for_iris_vllm_shutdown,
)
from marin.inference.vllm_server import IsolatedCudaVllm, VllmEnvironment, VllmLauncherWithEnvironment, VllmType

from experiments.evaluation.agentx import run as native
from experiments.evaluation.agentx.artifacts import copy_file
from experiments.evaluation.agentx.processes import run_logged
from experiments.evaluation.agentx.recipe import load_series, series_path
from experiments.evaluation.agentx.run import create_server, interrupt


def stage_candidate_wheel(source: dict) -> Path:
    directory = Path("/iris/candidates") / source["sha256"]
    directory.mkdir(parents=True, exist_ok=True)
    wheel = directory / source["name"]
    if not wheel.exists():
        copy_file(source["uri"], str(wheel))
    with wheel.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == source["sha256"]
    assert wheel.stat().st_size == source["bytes"]
    receipt = source["receipt"] | {"wheel_path": str(wheel)}
    assert receipt["wheel_sha256"] == source["sha256"]
    path = directory / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return path


def use_candidate_wheel(server: VllmEnvironment, receipt_file: Path, base: dict) -> dict:
    receipt = json.loads(receipt_file.read_text())
    wheel = Path(receipt["wheel_path"])
    with wheel.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == receipt["wheel_sha256"]
    assert receipt["base_wheel_sha256"] == next(
        item["sha256"] for item in base["wheels"] if item["architecture"] == "aarch64"
    )
    release = VllmGpuRelease(
        release_tag=f"private-hero-probe-{receipt['source_commit'][:12]}",
        source_commit=receipt["source_commit"],
        version=base["version"],
        torch_backend=base["torch_backend"],
        torch_version=base["torch_version"],
        wheels=(VllmGpuWheel("aarch64", ("10.0",), wheel.as_uri(), receipt["wheel_sha256"]),),
    )
    assert isinstance(server.launcher, VllmLauncherWithEnvironment)
    server.launcher = replace(server.launcher, launcher=IsolatedCudaVllm(source=VllmType.MARIN_FORK, release=release))
    return receipt


def use_numa_binding(server: VllmEnvironment, utility: dict) -> None:
    directory = Path("/iris/candidates") / ("numactl-" + utility["deb"]["sha256"])
    directory.mkdir(parents=True, exist_ok=True)
    package = directory / utility["deb"]["name"]
    copy_file(utility["deb"]["uri"], str(package))
    assert hashlib.sha256(package.read_bytes()).hexdigest() == utility["deb"]["sha256"]
    subprocess.run(["dpkg-deb", "--extract", str(package), str(directory / "root")], check=True)
    binary = directory / "root/usr/bin/numactl"
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == utility["binary_sha256"]
    server.launcher = VllmLauncherWithEnvironment(
        server.launcher, {"PATH": str(binary.parent) + os.pathsep + os.environ["PATH"]}
    )
    server.extra_cli_args.append("--numa-bind")


def serve(
    manifest: dict,
    output: Path,
    pipeline_stages: int,
    serve_duration: float,
    profile: bool,
    prefix_cache: bool,
    max_batched_tokens: int | None,
    max_num_seqs: int | None,
    profile_warmup_iterations: int,
    candidate_wheel: Path | None,
    moe_backend: str | None,
    all2all_backend: str | None,
    numa_bind: bool,
    api_server_count: int | None,
    profile_light: bool,
    run_controls: bool,
    master_port: int | None,
) -> None:
    recipe = (
        Path(__file__).with_name(manifest["series_filename"])
        if "series_filename" in manifest
        else series_path(manifest["profile"])
    )
    if "series_filename" in manifest:
        native.series_path = lambda profile: recipe
    assert hashlib.sha256(recipe.read_bytes()).hexdigest() == manifest["series_sha256"]
    series = load_series(recipe)
    hardware = series["profiles"][manifest["profile"]]
    devices = hardware["hosts"] * hardware["gpus"]
    assert devices in (16, 32)
    output.mkdir(parents=True, exist_ok=False)
    identity = {key: os.environ[key] for key in ("IRIS_TASK_ID", "IRIS_ATTEMPT_UID", "IRIS_NODE_NAME", "IRIS_BUNDLE_ID")}
    identity["captured_unix"] = time.time()
    identity["gpus"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True
    ).splitlines()
    (output / "hardware-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    if "candidate_wheel_input" in manifest:
        assert candidate_wheel is None
        candidate_wheel = stage_candidate_wheel(manifest["candidate_wheel_input"])
    record = {
        "source": manifest["source"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "series_sha256": manifest["series_sha256"],
        "iris_bundle_id": os.environ["IRIS_BUNDLE_ID"],
        "iris_task_id": os.environ["IRIS_TASK_ID"],
        "iris_attempt_uid": os.environ["IRIS_ATTEMPT_UID"],
        "iris_node_name": os.environ["IRIS_NODE_NAME"],
        "worker_pid": os.getpid(),
        "started_unix": time.time(),
        "requested_parallelism": {
            "tp": 1,
            "pp": pipeline_stages,
            "dp": devices // pipeline_stages,
            "ep": devices // pipeline_stages,
        },
        "physical_gpus": devices,
        "profiler_configured": profile,
        "overrides": {
            "prefix_cache": prefix_cache,
            "max_batched_tokens": max_batched_tokens,
            "max_num_seqs": max_num_seqs,
            "moe_backend": moe_backend,
            "all2all_backend": all2all_backend,
            "numa_bind": numa_bind,
            "api_server_count": api_server_count,
            "master_port": master_port,
        },
        "status": "starting",
    }
    record_path = output / "diagnostic.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    native_logs = None
    observer = None
    launch: IrisVllmLaunch | None = None
    try:
        observer = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("server_cpu.py")),
                "--output",
                str(output / "server-cpu.jsonl"),
                "--samples",
                "720",
            ]
        )
        with iris_vllm_launch(
            pipeline_parallel_size=pipeline_stages,
            tensor_parallel_size=1,
            data_parallel_size=devices // pipeline_stages,
            data_parallel_size_local=hardware["gpus"],
        ) as launch:
            server = create_server(manifest, output, launch)
            if candidate_wheel is not None:
                record["candidate_wheel"] = use_candidate_wheel(server, candidate_wheel, manifest["vllm_release"])
            if prefix_cache:
                server.extra_cli_args.remove("--no-enable-prefix-caching")
                server.extra_cli_args.append("--enable-prefix-caching")
            if "expert_load" in manifest.get("diagnostic_request", {}):
                server.extra_cli_args.append("--enable-return-routed-experts")
                record["overrides"]["return_routed_experts"] = True
            for flag, value in (
                ("--max-num-batched-tokens", max_batched_tokens),
                ("--max-num-seqs", max_num_seqs),
                ("--kernel-config", json.dumps({"moe_backend": moe_backend}) if moe_backend is not None else None),
            ):
                if value is not None:
                    server.extra_cli_args[server.extra_cli_args.index(flag) + 1] = str(value)
            for flag, value in (("--all2all-backend", all2all_backend), ("--api-server-count", api_server_count)):
                if value is not None:
                    server.extra_cli_args.extend([flag, str(value)])
            if numa_bind:
                use_numa_binding(server, manifest["numactl_input"])
                record["private_numactl"] = manifest["numactl_input"]
            memory_fraction = manifest.get("diagnostic_request", {}).get("gpu_memory_utilization")
            if memory_fraction is not None:
                server.extra_cli_args[server.extra_cli_args.index("--gpu-memory-utilization") + 1] = str(memory_fraction)
                record["overrides"]["gpu_memory_utilization"] = memory_fraction
            if master_port is not None:
                for flag, port in (("--master-port", master_port), ("--data-parallel-rpc-port", master_port + 1)):
                    if flag in server.extra_cli_args:
                        server.extra_cli_args[server.extra_cli_args.index(flag) + 1] = str(port)
            if profile:
                nccl_environment = {
                    "NCCL_DEBUG": "INFO",
                    "NCCL_DEBUG_SUBSYS": "INIT,GRAPH",
                    "NCCL_DEBUG_FILE": str(output / "nccl-%h-%p.log"),
                }
                server.launcher = VllmLauncherWithEnvironment(server.launcher, nccl_environment)
                record["nccl_environment"] = nccl_environment
                config = {
                    "profiler": "torch",
                    "torch_profiler_dir": str(output / "torch-profiler"),
                    "ignore_frontend": True,
                    "torch_profiler_with_stack": not profile_light,
                    "torch_profiler_record_shapes": not profile_light,
                    "torch_profiler_with_memory": False,
                    "delay_iterations": 2,
                    "warmup_iterations": profile_warmup_iterations,
                    "active_iterations": 5,
                    "max_iterations": 5,
                }
                server.extra_cli_args.extend(["--profiler-config", json.dumps(config)])
                record["profiler_config"] = config
            with server:
                assert server.vllm_server is not None
                native_logs = Path(server.vllm_server.log_dir)
                record["server_command"] = server.command
                record_path.write_text(json.dumps(record, indent=2) + "\n")
                if not launch.is_leader:
                    wait_for_iris_vllm_shutdown(launch, server.check_alive)
                    server.publish_compilation_cache()
                    record["status"] = "stopped"
                    return
                server.wait_until_ready()
                record.update(status="ready", ready_unix=time.time(), model_id=server.model_id)
                record_path.write_text(json.dumps(record, indent=2) + "\n")
                print("HERO_DIAGNOSTIC_READY", flush=True)
                deadline = time.monotonic() + serve_duration
                with iris_vllm_followers(launch):
                    if run_controls:
                        control_manifest = output / "control-manifest.json"
                        control_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
                        run_logged(
                            [
                                sys.executable,
                                str(
                                    Path(__file__).with_name(
                                        "expert_load.py"
                                        if "expert_load" in manifest.get("diagnostic_request", {})
                                        else "flag_controls.py"
                                    )
                                ),
                                "--manifest",
                                str(control_manifest),
                                "--output",
                                str(output),
                            ],
                            dict(os.environ),
                            output / "controls.log",
                            timeout=serve_duration,
                        )
                        (output / "stop").touch()
                    while not (output / "stop").exists() and time.monotonic() < deadline:
                        server.check_alive()
                        time.sleep(1)
                record["status"] = "stopped" if (output / "stop").exists() else "deadline_expired"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if observer is not None:
            observer.terminate()
            observer.wait(timeout=10)
        record["finished_unix"] = time.time()
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        if native_logs is not None:
            shutil.copytree(native_logs, output / "server-logs")
        if launch is not None and not launch.is_leader and record["status"] == "stopped":
            notify_iris_vllm_stopped(launch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pipeline-stages", type=int, choices=(1, 4), default=1)
    parser.add_argument("--serve-duration", type=float, default=1800)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--max-batched-tokens", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--profile-warmup-iterations", type=int, default=1)
    parser.add_argument("--candidate-wheel", type=Path)
    parser.add_argument("--moe-backend", choices=("triton", "auto", "flashinfer_trtllm", "flashinfer_cutlass"))
    parser.add_argument("--all2all-backend", choices=("allgather_reducescatter", "flashinfer_nvlink_one_sided"))
    parser.add_argument("--numa-bind", action="store_true")
    parser.add_argument("--api-server-count", type=int)
    parser.add_argument("--profile-light", action="store_true")
    parser.add_argument("--run-controls", action="store_true")
    parser.add_argument("--master-port", type=int)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    serve(
        json.loads(args.manifest.read_text()),
        args.output,
        args.pipeline_stages,
        args.serve_duration,
        args.profile,
        args.prefix_cache,
        args.max_batched_tokens,
        args.max_num_seqs,
        args.profile_warmup_iterations,
        args.candidate_wheel,
        args.moe_backend,
        args.all2all_backend,
        args.numa_bind,
        args.api_server_count,
        args.profile_light,
        args.run_controls,
        args.master_port,
    )


if __name__ == "__main__":
    main()
