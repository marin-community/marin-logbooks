# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diagnose a MoE graph boundary before changing full-Hero graph scheduling.

Run the retained tiny Hero on four local GPUs with native graph-input address
checks and synchronous CUDA error reporting. First retain global DP padding;
try per-rank padding only if that control survives. No throughput is measured.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
import urllib.request
from pathlib import Path

from cache_probe import batch_trace
from diagnostic_server import stage_candidate_wheel

from experiments.evaluation.agentx.processes import logged_process, stop_group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.output
    root.mkdir(exist_ok=False)
    worker_output = root / "workers"
    worker_output.mkdir()
    model = Path(manifest["model_dir"])
    with (model / "model.safetensors").open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == manifest["fixture_model_sha256"]
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    gpu_uuids = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).split()
    assert set(gpu_uuids) == set(manifest["gpu_uuids"])
    assert os.environ["IRIS_ATTEMPT_UID"] == manifest["attempt_uid"]
    receipt = json.loads(stage_candidate_wheel(manifest["candidate_wheel_input"]).read_text())
    prefix = manifest["serving_prefix"].copy()
    prefix[prefix.index("--from") + 1] = (
        "vllm @ " + Path(receipt["wheel_path"]).as_uri() + "#sha256=" + receipt["wheel_sha256"]
    )
    command = [
        *prefix,
        *manifest["cuda_bootstrap"],
        "python",
        "-m",
        "vllm.entrypoints.launchers.api_server.entry",
        *manifest["server_args"],
    ]
    environment = {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "UV_NO_CONFIG": "1",
        "UV_CONSTRAINT": "/app/experiments/evaluation/agentx/requirements-vllm-hero.txt",
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_LOGGING_LEVEL": "DEBUG",
        "CUDA_LAUNCH_BLOCKING": "1",
        "VLLM_CACHE_ROOT": manifest["cache_root"],
        "TORCHINDUCTOR_CACHE_DIR": manifest["cache_root"] + "/inductor",
        "TRITON_CACHE_DIR": manifest["cache_root"] + "/triton",
        "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent), os.environ.get("PYTHONPATH", "")]),
        "IRIS_OUTPUT_DIR": str(worker_output),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    base_url = manifest["base_url"]
    record = {
        "status": "starting",
        "scope": manifest["scope"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "command": command,
        "environment_overrides": environment,
        "attempt_uid": os.environ["IRIS_ATTEMPT_UID"],
        "gpu_uuids": gpu_uuids,
        "started_unix": time.time(),
        "phases": [],
    }
    server = None
    try:
        server = logged_process(command, dict(os.environ) | environment, root / "server.log")
        deadline = time.monotonic() + manifest.get("readiness_timeout_seconds", 600)
        while True:
            assert server.poll() is None, "Tiny server exits before readiness"
            assert time.monotonic() < deadline, "Tiny readiness deadline"
            try:
                with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            time.sleep(1)
        record.update(status="ready", ready_unix=time.time())
        (root / "progress.json").write_text(json.dumps(record, indent=2) + "\n")
        for local in manifest.get("local_modes", [False, True]):
            disable_replay = manifest.get("disable_graph_replay", False)
            phase = {
                "local_piecewise_dp": local,
                "disable_graph_replay": disable_replay,
                "requests": [],
                "status": "running",
            }
            record["phases"].append(phase)
            batch_trace(base_url, "install_hero_batch_trace", [None, disable_replay, local, True])
            phase["cache_storage"] = batch_trace(base_url, "hero_cache_storage")
            if manifest.get("attention_snapshot"):
                phase["attention_snapshot"] = batch_trace(
                    base_url, "install_hero_attention_snapshot", [manifest.get("attention_prefill_captures", 1)]
                )
            if manifest.get("moe_snapshot", False):
                assert disable_replay
                phase["moe_snapshot_installation"] = batch_trace(base_url, "install_hero_moe_snapshot")
            cases = manifest.get("requests")
            if "request_inputs" in manifest:
                source = Path(manifest["request_inputs"]["path"])
                assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["request_inputs"]["sha256"]
                inputs = json.loads(source.read_text())["prompts"]
                cases = [case | {"prompts": [inputs[index] for index in case["branches"]]} for case in cases]
            if cases is None:
                cases = [
                    {
                        "prompts": [[73 + i % 17 for i in range(length)] for length in lengths],
                        "cache_namespace": str(index),
                    }
                    for index, lengths in enumerate([[80], [17, 33], [15, 47, 63], [16]])
                ]
            for case in cases:
                prompts = case["prompts"]
                lengths = [len(prompt) for prompt in prompts]
                payload = {
                    "model": "tiny-hero-boundary",
                    "prompt": prompts,
                    "max_tokens": 16,
                    "min_tokens": 16,
                    "ignore_eos": True,
                    "temperature": 0,
                    "seed": 42,
                    "logprobs": 20,
                    "return_tokens_as_token_ids": True,
                    "cache_salt": f"boundary-{local}-{case['cache_namespace']}",
                }
                request = urllib.request.Request(
                    base_url + "/v1/completions",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json", "X-data-parallel-rank": "0"},
                )
                pending = {"input_lengths": lengths, "request": payload, "started_unix": time.time()}
                phase["requests"].append(pending)
                (root / "progress.json").write_text(json.dumps(record, indent=2) + "\n")
                with urllib.request.urlopen(request, timeout=60) as response:
                    pending["response"] = json.load(response)
                pending["finished_unix"] = time.time()
                assert len(pending["response"]["choices"]) == len(prompts)
                assert all(
                    math.isfinite(value)
                    for choice in pending["response"]["choices"]
                    for position in choice["logprobs"]["top_logprobs"]
                    for value in position.values()
                )
            phase.update(status="completed", traces=batch_trace(base_url, "take_hero_batch_trace"))
            batch_trace(base_url, "stop_hero_batch_trace")
            (root / "progress.json").write_text(json.dumps(record, indent=2) + "\n")
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="observed_failure", error_type=type(error).__name__, error=str(error))
        if record["phases"]:
            record["phases"][-1]["status"] = "failed"
        (root / "client-error.log").write_text(traceback.format_exc())
    finally:
        if server is not None:
            record["server_returncode_before_stop"] = server.poll()
            stop_group(server)
        record["finished_unix"] = time.time()
        (root / "progress.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({key: record[key] for key in ["status", "finished_unix", "phases"]}), flush=True)


if __name__ == "__main__":
    main()
