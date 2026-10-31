# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Use native AgentX orchestration with recorded serving overrides.

The manifest pins the unchanged base recipe/runtime and separately identifies
the effective prefix-cache and private-wheel overrides. Replay, preparation,
warmup, benchmark duration and all native validity/attempt fences stay native.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import zstandard
from cache_probe import batch_trace
from completion_retention import retain_finished_attempt
from diagnostic_server import (
    VllmLauncherWithEnvironment,
    prefetch_flashinfer_headers,
    set_compilation_config,
    stage_candidate_wheel,
    use_candidate_wheel,
    use_numa_binding,
)
from flag_controls import assert_exact_cross, prepare_tokenizer
from iris.cluster.client.job_info import get_job_info
from marin.inference.iris_vllm import IrisVllmLaunch
from marin.inference.vllm_server import VllmEnvironment
from router_process import upstream_router

from experiments.evaluation.agentx import run as native
from experiments.evaluation.agentx.artifacts import file_digest
from experiments.evaluation.agentx.recipe import PointManifest

BASE_CREATE_SERVER = native.create_server
BASE_WORKLOAD_ENVIRONMENT = native.workload_environment
BASE_BENCHMARK_COMMAND = native.benchmark_command
BASE_RUN_LOGGED = native.run_logged
SESSION_HEADER_ENV = "AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID"


def affinity_environment(bundle: Path, series: dict) -> dict[str, str]:
    return BASE_WORKLOAD_ENVIRONMENT(bundle, series) | {SESSION_HEADER_ENV: "1"}


def create_candidate_server(
    manifest: PointManifest, output: Path, launch: IrisVllmLaunch | None = None
) -> VllmEnvironment:
    overrides = manifest["runtime_overrides"]
    assert overrides["enable_prefix_caching"] is True
    server = BASE_CREATE_SERVER(manifest, output, launch)
    receipt = None
    if "candidate_receipt" in overrides:
        receipt = use_candidate_wheel(server, Path(overrides["candidate_receipt"]), manifest["vllm_release"])
        assert receipt["wheel_sha256"] == overrides["wheel_sha256"]
        assert receipt["source_commit"] == overrides["source_commit"]
    server.extra_cli_args.remove("--no-enable-prefix-caching")
    server.extra_cli_args.append("--enable-prefix-caching")
    for key, flag in (
        ("max_num_batched_tokens", "--max-num-batched-tokens"),
        ("master_port", "--master-port"),
        ("gpu_memory_utilization", "--gpu-memory-utilization"),
    ):
        if key in overrides:
            server.extra_cli_args[server.extra_cli_args.index(flag) + 1] = str(overrides[key])
    if "master_port" in overrides and "--data-parallel-rpc-port" in server.extra_cli_args:
        server.extra_cli_args[server.extra_cli_args.index("--data-parallel-rpc-port") + 1] = str(
            overrides["master_port"] + 1
        )
    if overrides.get("numa_bind"):
        assert overrides["numa_bind"] is True
        use_numa_binding(server, manifest["numactl_input"])
    if "compilation_config" in overrides:
        set_compilation_config(server, overrides["compilation_config"])
    if "additional_config" in overrides:
        server.extra_cli_args.extend(["--additional-config", json.dumps(overrides["additional_config"])])
    if "kernel_config" in overrides:
        index = server.extra_cli_args.index("--kernel-config") + 1
        kernel = json.loads(server.extra_cli_args[index]) | overrides["kernel_config"]
        assert kernel["moe_backend"] == "triton"
        server.extra_cli_args[index] = json.dumps(kernel)
    environment = dict(overrides.get("environment_overrides", {}))
    if manifest.get("startup_numerics"):
        assert server.host == "127.0.0.1"
        server.extra_cli_args.extend(["--worker-extension-cls", "tiny_graph_probe.BatchTraceWorkerExtension"])
        environment.update(
            VLLM_SERVER_DEV_MODE="1",
            PYTHONPATH=os.pathsep.join([str(Path(__file__).parent), os.environ.get("PYTHONPATH", "")]),
        )
    if environment:
        server.launcher = VllmLauncherWithEnvironment(server.launcher, environment)
    if overrides.get("prefetch_flashinfer_headers"):
        prefetch_flashinfer_headers(server, output)
    (output / "runtime-overrides.json").write_text(
        json.dumps(
            {
                "requested": overrides,
                "wheel": receipt,
                "client_environment": {SESSION_HEADER_ENV: "1"} if overrides.get("session_affinity") else {},
            },
            indent=2,
        )
        + "\n"
    )
    return server


def run_startup_numerics(manifest: dict, output: Path) -> None:
    """Check this fresh server before the unchanged AgentX warmup starts."""
    control = manifest["startup_numerics"]
    reference = Path(control["source_reference"]["path"])
    assert (
        hashlib.sha256((reference.parent / "control-manifest.json").read_bytes()).hexdigest()
        == control["source_reference"]["manifest_sha256"]
    )
    assert json.loads((reference.parent / "diagnostic.json").read_text())["status"] == "stopped"
    root = output / "startup-numerics"
    root.mkdir(exist_ok=False)
    started = time.time()
    actual = json.loads((output / "attempt.json").read_text())
    assert actual["status"] == "measuring"
    command = actual["server_command"]
    overrides = manifest["runtime_overrides"]
    assert json.loads(command[command.index("--compilation-config") + 1]) == overrides["compilation_config"]
    if "additional_config" in overrides:
        assert json.loads(command[command.index("--additional-config") + 1]) == overrides["additional_config"]
    kernel = json.loads(command[command.index("--kernel-config") + 1])
    assert all(kernel[key] == value for key, value in overrides["kernel_config"].items())
    prefix = [
        "env",
        "UV_NO_CONFIG=1",
        "UV_CONSTRAINT=/app/experiments/evaluation/agentx/requirements-vllm-hero.txt",
        *command[: command.index("python")],
    ]
    assert "uvx" in prefix and "--from" in prefix
    base_url = manifest["router"]["backend_url"]
    ready = batch_trace(base_url, "capture_hero_compiler_sources", ["ready"])
    (root / "compiler-sources-ready.json").write_text(json.dumps(ready, indent=2) + "\n")
    tokenizer = prepare_tokenizer(manifest, root / "tokenizer")
    probe = [
        *prefix,
        "python",
        str(Path(__file__).with_name("cache_probe.py")),
        "--tokenizer",
        str(tokenizer / "tokenizer.json"),
        "--base-url",
        base_url,
        "--rank",
        "0",
        "--trace-batches",
        "--compare-to",
        str(reference),
        "--output",
        str(root / "probe"),
    ]
    BASE_RUN_LOGGED(probe, dict(os.environ), root / "probe.log", timeout=900)
    checks = json.loads((root / "probe/comparison.json").read_text())["comparisons"]
    assert len(checks) == 8 and all(
        row["same_tokens"] and row["finite_logprobs"] and row["max_shared_logprob_absolute_delta"] == 0 for row in checks
    )
    assert_exact_cross(root / "probe/cross-comparison.json")
    after = batch_trace(base_url, "capture_hero_compiler_sources", ["after-numerics"])
    (root / "compiler-sources-after.json").write_text(json.dumps(after, indent=2) + "\n")
    request = urllib.request.Request(base_url + "/reset_prefix_cache", data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=90) as response:
        reset = json.load(response)
    assert reset == {"success": True}
    record = {
        "status": "passed_before_agentx_warmup",
        "started_unix": started,
        "finished_unix": time.time(),
        "source_reference": control["source_reference"],
        "prefix_cache_reset": reset,
        "scope": (
            "Ten fixed-history source waves and eight state pairs on the actual fresh server, "
            "outside the unchanged AgentX warmup/sending/drain window. "
            "Native cache reset broadcasts to all engines and reports its first engine result."
        ),
    }
    (root / "result.json").write_text(json.dumps(record, indent=2) + "\n")


def pack_metrics_export(output: Path) -> None:
    source = output / "point/aiperf_artifacts/server_metrics_export.json"
    if not source.exists() or source.stat().st_size < 512 * 1024**2:
        return
    compressed = source.with_suffix(".json.zst")
    expected = file_digest(source)
    size = source.stat().st_size
    with source.open("rb") as reader, compressed.open("wb") as writer:
        with zstandard.ZstdCompressor().stream_writer(writer) as encoded:
            shutil.copyfileobj(reader, encoded, length=8 * 1024**2)
    with compressed.open("rb") as encoded, zstandard.ZstdDecompressor().stream_reader(encoded) as decoded:
        assert hashlib.file_digest(decoded, "sha256").hexdigest() == expected
    receipt = {
        "raw_sha256": expected,
        "raw_bytes": size,
        "compressed_sha256": file_digest(compressed),
        "compressed_bytes": compressed.stat().st_size,
        "codec": "zstandard",
        "restore": "Decompress server_metrics_export.json.zst; verify raw_sha256 before processing.",
    }
    compressed.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    source.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("startup_numerics"):
        assert "router" in manifest
    if "series_filename" in manifest:
        path = Path(__file__).with_name(manifest["series_filename"])
        native.series_path = lambda profile: path
    if "candidate_wheel_input" in manifest:
        receipt = stage_candidate_wheel(manifest["candidate_wheel_input"])
        assert str(receipt) == manifest["runtime_overrides"]["candidate_receipt"]
    native.create_server = create_candidate_server
    if manifest["runtime_overrides"].get("session_affinity"):
        native.workload_environment = affinity_environment
    if "router" in manifest:
        assert manifest["runtime_overrides"]["session_affinity"] is True

        def router_benchmark_command(series: dict, concurrency: int, url: str, output: Path) -> list[str]:
            command = BASE_BENCHMARK_COMMAND(series, concurrency, url, output)
            router_url = manifest["router"]["base_url"]
            command[command.index("--url") + 1] = router_url
            # AIPerf also discovers --url/metrics. Use that same endpoint once.
            command[command.index("--server-metrics") + 1] = f"{router_url}/metrics"
            return command

        def run_with_router(command: list[str], environment: dict, log: Path, *, timeout: int) -> None:
            if "agentx" not in command or "profile" not in command:
                BASE_RUN_LOGGED(command, environment, log, timeout=timeout)
                return
            # The pinned warmup is count driven, so its grace setting is not
            # a wall-clock bound. Use the recipe's existing task limit.
            series = native.load_series(native.series_path(manifest["profile"]))
            process_timeout = series["task_timeout"]
            assert process_timeout >= timeout
            (log.parent / "client-process-timeout.json").write_text(
                json.dumps({"native_requested": timeout, "effective": process_timeout, "source": "series.task_timeout"})
                + "\n"
            )
            if manifest.get("startup_numerics"):
                run_startup_numerics(manifest, log.parent)
            with upstream_router(manifest["router"], log.parent):
                BASE_RUN_LOGGED(command, environment, log, timeout=process_timeout)

        native.benchmark_command = router_benchmark_command
        native.run_logged = run_with_router
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    observer = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("server_cpu.py")),
            "--output",
            str(output / "server-cpu.jsonl"),
            "--samples",
            "3600",
        ]
    )
    monitor = None
    job_info = get_job_info()
    assert job_info is not None
    if job_info.task_index == 0:
        monitor = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("client_load.py")),
                "--output",
                str(output / "client-load.jsonl"),
                "--duration",
                "10800",
            ]
        )
    try:
        native.run(manifest)
    finally:
        observer.terminate()
        observer.wait(timeout=10)
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)
        try:
            if job_info.task_index == 0:
                pack_metrics_export(output)
        finally:
            if prefix := manifest.get("completion_retention_prefix"):
                try:
                    receipt = retain_finished_attempt(
                        output, prefix, job_info.task_index, os.environ["IRIS_ATTEMPT_UID"]
                    )
                    print("AGENTX_FINISHED_OUTPUT_RETAINED", json.dumps(receipt), flush=True)
                except Exception as error:
                    # Preserve the native application outcome and normal Iris
                    # capture. This receipt must exist before claiming retention.
                    print("AGENTX_FINISHED_OUTPUT_RETENTION_FAILED", type(error).__name__, flush=True)


if __name__ == "__main__":
    main()
