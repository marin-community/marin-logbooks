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
from marin.inference.vllm_server import IsolatedCudaVllm, VllmLauncherWithEnvironment, VllmType

from experiments.evaluation.agentx.recipe import load_series, series_path
from experiments.evaluation.agentx.run import create_server, interrupt


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
) -> None:
    recipe = series_path(manifest["profile"])
    assert hashlib.sha256(recipe.read_bytes()).hexdigest() == manifest["series_sha256"]
    series = load_series(recipe)
    hardware = series["profiles"][manifest["profile"]]
    devices = hardware["hosts"] * hardware["gpus"]
    assert devices == 16
    output.mkdir(parents=True, exist_ok=False)
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
        },
        "status": "starting",
    }
    record_path = output / "diagnostic.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    native_logs = None
    launch: IrisVllmLaunch | None = None
    try:
        with iris_vllm_launch(
            pipeline_parallel_size=pipeline_stages,
            tensor_parallel_size=1,
            data_parallel_size=devices // pipeline_stages,
            data_parallel_size_local=hardware["gpus"],
        ) as launch:
            server = create_server(manifest, output, launch)
            if candidate_wheel is not None:
                receipt = json.loads(candidate_wheel.read_text())
                wheel = Path(receipt["wheel_path"])
                with wheel.open("rb") as stream:
                    assert hashlib.file_digest(stream, "sha256").hexdigest() == receipt["wheel_sha256"]
                base = manifest["vllm_release"]
                assert receipt["base_wheel_sha256"] == next(
                    item["sha256"] for item in base["wheels"] if item["architecture"] == "aarch64"
                )
                release = VllmGpuRelease(
                    release_tag="private-shortconv-output-buffer",
                    source_commit=receipt["source_commit"],
                    version=base["version"],
                    torch_backend=base["torch_backend"],
                    torch_version=base["torch_version"],
                    wheels=(VllmGpuWheel("aarch64", ("10.0",), wheel.as_uri(), receipt["wheel_sha256"]),),
                )
                assert isinstance(server.launcher, VllmLauncherWithEnvironment)
                server.launcher = replace(
                    server.launcher, launcher=IsolatedCudaVllm(source=VllmType.MARIN_FORK, release=release)
                )
                record["candidate_wheel"] = receipt
            if prefix_cache:
                server.extra_cli_args.remove("--no-enable-prefix-caching")
                server.extra_cli_args.append("--enable-prefix-caching")
            for flag, value in (
                ("--max-num-batched-tokens", max_batched_tokens),
                ("--max-num-seqs", max_num_seqs),
            ):
                if value is not None:
                    server.extra_cli_args[server.extra_cli_args.index(flag) + 1] = str(value)
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
                    "torch_profiler_with_stack": True,
                    "torch_profiler_record_shapes": True,
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
                    while not (output / "stop").exists() and time.monotonic() < deadline:
                        server.check_alive()
                        time.sleep(1)
                record["status"] = "stopped" if (output / "stop").exists() else "deadline_expired"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
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
    )


if __name__ == "__main__":
    main()
