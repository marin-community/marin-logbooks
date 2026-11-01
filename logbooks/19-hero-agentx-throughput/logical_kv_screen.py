# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one synthetic screen on a fresh, numerically qualified native16 server."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from flag_controls import prepare_tokenizer
from iris.cluster.client.job_info import get_job_info
from replica_agentx import serving_attempts
from replica_screen import metrics

from experiments.evaluation.agentx.processes import run_logged

CONTROL_ONLY_FIELDS = frozenset(
    {
        "run_controls",
        "numerics_only",
        "scope",
        "master_port",
        "source_reference",
        "additional_cases",
        "capture_compiler_sources",
        "trace_control_batches",
    }
)


def model_controls(request: dict) -> dict:
    return {key: value for key, value in request.items() if key not in CONTROL_ONLY_FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    qualification_bytes = args.qualification.read_bytes()
    assert hashlib.sha256(qualification_bytes).hexdigest() == manifest["qualification_sha256"]
    qualification = json.loads(qualification_bytes)
    assert qualification["full_model_qualified"] and qualification["all8_raw_readbacks_passed"]
    assert qualification["all24_state_pairs_exact_each_mode"] and qualification["all30_source_waves_exact"]
    assert qualification["wheel_sha256"] == manifest["candidate_wheel_input"]["sha256"]
    assert qualification["compiler_config"] == manifest["diagnostic_request"]["compilation_config"]
    assert qualification["series_sha256"] == manifest["series_sha256"]
    assert qualification["vllm_release"] == manifest["vllm_release"]
    assert qualification["workload_key"] == manifest["workload"]["key"]
    mode = "on" if manifest["diagnostic_request"]["additional_config"]["grug_logical_kv_cache"] else "off"
    assert manifest["qualified_manifest_sha256"] == qualification["modes"][mode]["manifest_sha256"]
    assert model_controls(manifest["diagnostic_request"]) == qualification["modes"][mode]["model_controls"]
    info = get_job_info()
    assert info is not None and info.num_tasks == 4 and info.task_index == 0
    assert str(info.job_id) == manifest["native_job_id"]
    primary = json.loads(args.diagnostic.read_text())
    assert primary["status"] == "ready" and primary["physical_gpus"] == 16
    assert primary["started_unix"] > qualification["qualified_unix"]
    assert primary["candidate_wheel"]["wheel_sha256"] == qualification["wheel_sha256"]
    server_manifest = json.loads(args.diagnostic.with_name("input-manifest.json").read_text())
    assert server_manifest == manifest
    assert primary["requested_parallelism"] == {"tp": 1, "pp": 1, "dp": 16, "ep": 16}
    backend = {
        "job_id": str(info.job_id),
        "dp": 16,
        "url": "http://127.0.0.1:38197",
        "gang_attempts": manifest["native_controller_attempts"],
    }
    attempts = serving_attempts([backend])
    assert attempts == manifest["native_controller_attempts"]
    command = primary["server_command"]
    prefix = ["env", "UV_NO_CONFIG=1", "UV_CONSTRAINT=/app/experiments/evaluation/agentx/requirements-vllm-hero.txt"]
    prefix += command[: command.index("python")]
    assert "uvx" in prefix and "--from" in prefix
    with args.diagnostic.with_name("synthetic-client-claim.json").open("x") as stream:
        json.dump({"mode": mode, "pid": os.getpid(), "started_unix": time.time()}, stream)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_bytes(args.manifest.read_bytes())
    (args.output / "qualification.json").write_bytes(qualification_bytes)
    record = {
        "status": "starting",
        "mode": mode,
        "physical_gpus": 16,
        "layout": [16],
        "native_controller_attempts": attempts,
        "started_unix": time.time(),
        "scope": "One fresh synthetic HTTP workload. No AgentX throughput claim.",
    }
    try:
        tokenizer = prepare_tokenizer(manifest, args.output / "tokenizer")
        before_dir = args.output / "before"
        before_dir.mkdir()
        before = metrics([backend], before_dir)
        assert all(
            row.get("vllm:request_success_total", 0) == 0 for ranks in before.values() for row in ranks.values()
        ), "Fresh server has prior requests"
        case = manifest["screen_case"]
        argv = [*prefix, *case["argv"], "--result-dir", str(args.output), "--result-filename", "benchmark.json"]
        argv[argv.index("--tokenizer") + 1] = str(tokenizer)
        (args.output / "command.json").write_text(json.dumps(argv, indent=2) + "\n")
        record["status"] = "measuring"
        (args.output / "attempt.json").write_text(json.dumps(record, indent=2) + "\n")
        run_logged(argv, dict(os.environ), args.output / "client.log", timeout=case["timeout_seconds"])
        result = json.loads((args.output / "benchmark.json").read_text())
        assert result["failed"] == 0 and result["completed"] == 256
        assert result["total_output_tokens"] == 65536
        assert serving_attempts([backend]) == attempts
        after_dir = args.output / "after"
        after_dir.mkdir()
        metrics([backend], after_dir)
        record.update(status="client_completed_pending_external_native_attempt_fences_and_retention", benchmark=result)
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["finished_unix"] = time.time()
        (args.output / "attempt.json").write_text(json.dumps(record, indent=2) + "\n")
        args.diagnostic.with_name("stop").touch(exist_ok=False)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
