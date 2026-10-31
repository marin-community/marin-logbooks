# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the unchanged AgentX hour through qualified native replica groups."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.resources.state import JobState, TaskState
from replica_screen import completed_jobs_before_launch, metrics
from router_process import upstream_router

from experiments.evaluation.agentx.processes import run_logged
from experiments.evaluation.agentx.recipe import benchmark_command, load_series, workload_environment
from experiments.evaluation.agentx.workload import check_native_cache, restore_workload

SESSION_HEADER_ENV = "AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID"


def replica_benchmark_command(
    series: dict, concurrency: int, router_url: str, backends: list[dict], output: Path
) -> list[str]:
    """Keep native replay and scoring arguments; add each real metrics endpoint."""
    command = benchmark_command(series, concurrency, router_url, output)
    index = command.index("--server-metrics") + 1
    command[index : index + 1] = [backend["url"] + "/metrics" for backend in backends]
    return command


def serving_attempts(backends: list[dict]) -> dict[str, str]:
    """Read each actual native job; independent groups retain their own JobInfo."""
    client = iris_ctx().client
    assert client is not None
    result = {}
    for backend in backends:
        jobs = [
            job for job in client.list_jobs(prefix=backend["job_id"], limit=20) if str(job.job_id) == backend["job_id"]
        ]
        assert len(jobs) == 1
        job = client.job_status(jobs[0].job_id)
        assert job.state == JobState.RUNNING
        tasks = client.list_tasks(job.job_id)
        assert len(tasks) * 4 == backend["dp"]
        actual = {}
        for task in tasks:
            assert task.state == TaskState.RUNNING
            attempt = next(a for a in task.attempts if a.attempt_number == task.current_attempt_number)
            assert attempt.state == TaskState.RUNNING
            actual[str(task.task_id)] = attempt.attempt_uid
        assert actual == backend["gang_attempts"]
        assert result.keys().isdisjoint(actual)
        result.update(actual)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-diagnostic", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    series = load_series(args.series)
    assert hashlib.sha256(args.series.read_bytes()).hexdigest() == manifest["series_sha256"]
    assert series["duration"] == 3600 and series["sessions"] == 393 and series["max_context"] == 262144
    assert series["harness_version"] == "1.0.12" and series["random_seed"] == 42
    backends = manifest["replica_backends"]
    assert [backend["dp"] for backend in backends] in ([32], [16, 16])
    assert len({backend["url"] for backend in backends}) == len(backends)
    info = get_job_info()
    assert info is not None and info.task_index == 0
    assert str(info.job_id) == backends[0]["job_id"] and info.num_tasks * 4 == backends[0]["dp"]
    primary = json.loads(args.primary_diagnostic.read_text())
    assert primary["status"] == "ready" and primary["api_bind"]["url"] == backends[0]["url"]
    assert primary["gang_attempts"] == backends[0]["gang_attempts"] and not primary["profiler_configured"]
    qualification_bytes = args.qualification.read_bytes()
    assert hashlib.sha256(qualification_bytes).hexdigest() == manifest["qualification_input"]["sha256"]
    qualification = json.loads(qualification_bytes)
    assert qualification["qualified"] and qualification["all_native_attempt_fences_passed"]
    assert qualification["all_groups_stopped"] and qualification["layout"] == [backend["dp"] for backend in backends]
    assert qualification["server_recipe_sha256"] == manifest["server_recipe_sha256"]
    assert qualification["retention"] and all(
        row["archive_sha256"] == row["readback_sha256"] for row in qualification["retention"]
    )
    attempts = serving_attempts(backends)
    assert set(attempts.values()).isdisjoint(qualification["gang_attempts"].values())
    checks = completed_jobs_before_launch(manifest.get("completed_jobs_before_launch", []))
    with args.primary_diagnostic.with_name("replica-client-claim.json").open("x") as stream:
        json.dump({"phase": "agentx", "pid": os.getpid(), "started_unix": time.time()}, stream)
    args.output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "preparing",
        "source_commit": manifest["source_commit"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "server_recipe_sha256": manifest["server_recipe_sha256"],
        "layout": [backend["dp"] for backend in backends],
        "physical_gpus": sum(backend["dp"] for backend in backends),
        "primary_native_num_tasks": info.num_tasks,
        "gang_attempts": attempts,
        "prelaunch_job_checks": checks,
        "started_unix": time.time(),
        "scope": "Native AgentX client; success still requires external provider fences and raw retention.",
    }
    record_path = args.output / "attempt.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    monitor = None
    try:
        bundle = args.output / "bundle"
        restore_workload(manifest["workload"], bundle)
        check_native_cache(bundle, args.series, args.output / "workload-check.log")
        environment = workload_environment(bundle, series) | {SESSION_HEADER_ENV: "1"}
        router = manifest["router"] | {
            "backend_urls": [backend["url"] for backend in backends],
            "data_parallel_size": backends[0]["dp"],
        }
        router.pop("backend_url", None)
        command = replica_benchmark_command(
            series, manifest["concurrency"], router["base_url"], backends, args.output / "aiperf_artifacts"
        )
        (args.output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        (args.output / "metrics-endpoint-policy.json").write_text(
            json.dumps(
                {
                    "included_native_endpoints": [backend["url"] + "/metrics" for backend in backends],
                    "excluded_router_alias": router["base_url"] + "/metrics",
                    "reason": "Released Router proxies one worker; it does not aggregate independent replicas.",
                    "identity": (
                        "Keep endpoint plus engine rank; identical rank labels in different groups are distinct."
                    ),
                    "client_scoring": "Native AgentX scoring and validity stay unchanged.",
                },
                indent=2,
            )
            + "\n"
        )
        before = args.output / "before"
        before.mkdir()
        metrics(backends, before)
        monitor = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("client_load.py")),
                "--output",
                str(args.output / "client-load.jsonl"),
                "--duration",
                str(series["task_timeout"]),
            ]
        )
        with upstream_router(router, args.output):
            assert serving_attempts(backends) == attempts
            record.update({"status": "measuring", "client_started_unix": time.time()})
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            # The native preset owns its count-driven warmup and 3600s send /
            # 40s drain. Its grace setting is not a wall-clock warmup limit.
            run_logged(command, environment, args.output / "agentx.log", timeout=series["task_timeout"])
            exported = json.loads((args.output / "aiperf_artifacts/profile_export_aiperf.json").read_text())
            assert exported.get("metadata", {}).get("submission_valid") and not exported.get("was_cancelled")
            assert serving_attempts(backends) == attempts
        after = args.output / "after"
        after.mkdir()
        metrics(backends, after)
        record["status"] = "client_completed_pending_external_native_attempt_fences_and_retention"
    except BaseException as error:
        record.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        record["finished_unix"] = time.time()
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=15)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
