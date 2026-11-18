# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deploy a pinned serving profile and run the shared pytest adoption gate."""

import argparse
import json
import os
import time
from pathlib import Path

from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import CoschedulingConfig, Entrypoint, EnvironmentSpec, JobName, ResourceSpec, gpu_device
from iris.rpc import job_pb2
from marin.testing.inference.goldens import LookupStatus, lookup_reference, read_fixture
from rigging.filesystem.storage_path import prefix_join
from rigging.timing import Duration

from experiments.evaluation.knotmeter.allocation import allocation_cost, track_allocations
from experiments.evaluation.knotmeter.artifacts import retain_artifacts, write_record
from experiments.evaluation.knotmeter.cli import resolve_plan
from experiments.evaluation.knotmeter.config import Runtime, load_profile
from experiments.evaluation.knotmeter.processes import run_logged

DEFAULT_TEST_NODE = "tests/cluster/vllm/test_inference_correctness.py"
REMOTE_TEST_FILES = (
    "tests/__init__.py",
    "tests/conftest.py",
    "tests/cluster/__init__.py",
    "tests/cluster/conftest.py",
    "tests/cluster/vllm/__init__.py",
    "tests/cluster/vllm/conftest.py",
    DEFAULT_TEST_NODE,
)


def run_pytest_gate(command: list[str], plan: dict) -> None:
    """Retain pytest's failure details after endpoint fixture cleanup finishes."""
    info = get_job_info()
    if info is None:
        raise RuntimeError("The deployment gate requires Iris job context")
    output = Path(os.environ["IRIS_OUTPUT_DIR"]) / "pytest"
    output.mkdir(parents=True, exist_ok=False)
    Path("correctness-plan.json").write_text(json.dumps(plan))
    prefix = prefix_join(plan["trial_uri"], f"attempts/{os.environ['IRIS_ATTEMPT_UID']}/hosts/{info.task_index}")
    try:
        run_logged(
            [*command, "--junitxml", str(output / "pytest.xml")],
            dict(os.environ),
            output / "pytest.log",
            plan["series"]["task_timeout"],
        )
    finally:
        uri, digest = retain_artifacts(output, output.parent / "pytest.tar.zst", prefix, name="pytest")
        write_record(
            prefix_join(prefix, "pytest-receipt.json"),
            {
                "archive_uri": uri,
                "archive_sha256": digest,
                "source": plan["source"],
                "task_index": info.task_index,
                "attempt_uid": os.environ["IRIS_ATTEMPT_UID"],
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--runtime-manifest", help="Published release or normalized experimental build manifest")
    parser.add_argument("--set", action="append", default=[], help="The same YAML-valued overrides as perfmax")
    parser.add_argument(
        "--output", type=Path, required=True, help="Local execution receipt; includes the durable run URI"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--pytest-node", default=DEFAULT_TEST_NODE, help="Override the full gate with a focused diagnostic pytest node"
    )
    args = parser.parse_args()
    profile = load_profile(args.profile, args.set)
    if args.runtime_manifest:
        profile = profile.model_copy(update={"vllm": Runtime(manifest=args.runtime_manifest)})
    for check in profile.checks:
        if check.fixture:
            found = lookup_reference(read_fixture(check.fixture))
            if found.status != LookupStatus.FOUND:
                raise ValueError(f"Required reference is {found.status}: {found.reason}")
    plan = resolve_plan(profile, args.name)
    plan["series"] |= {"scheduling_timeout": 1800, "task_timeout": 10800}
    plan["pytest_node"] = args.pytest_node
    plan["diagnostic"] = args.pytest_node != DEFAULT_TEST_NODE
    plan["trial_uri"] = prefix_join(plan["marin_prefix"], f"inference/correctness/runs/{plan['trial_id']}")
    plan["gpu_job_id"] = plan["job_id"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2))
    print(
        json.dumps(
            {
                "job_id": plan["job_id"],
                "cluster": plan["cluster"],
                "run_uri": plan["trial_uri"],
                "gpus": profile.gpus.model_dump(),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return
    command = [
        "uv",
        "run",
        "--frozen",
        "--package",
        "marin-core",
        "--group",
        "test",
        "--extra",
        "cpu",
        "pytest",
        args.pytest_node,
        "-m",
        "cluster",
        "-o",
        "addopts=",
        "--import-mode=importlib",
        "--correctness-plan",
        "correctness-plan.json",
        "-v",
        "-s",
    ]
    with connect_controller(cluster_name=plan["cluster"]) as endpoint:
        with IrisClient.remote(
            endpoint.url,
            credentials=endpoint.credentials,
            workspace=Path.cwd(),
            extra_bundle_includes=tuple(dict.fromkeys((*REMOTE_TEST_FILES, args.pytest_node.partition("::")[0]))),
        ) as client:
            job = client.submit(
                entrypoint=Entrypoint.from_callable(run_pytest_gate, command, plan),
                name=JobName.from_string(plan["job_id"]).name,
                resources=ResourceSpec(
                    cpu=profile.cpu,
                    memory=profile.memory,
                    disk=profile.disk,
                    device=gpu_device(profile.gpus.type, profile.gpus.per_host),
                ),
                environment=EnvironmentSpec(
                    sync_packages=["marin-core"],
                    extras=["cpu"],
                    env_vars={"MARIN_PREFIX": plan["marin_prefix"], "JAX_PLATFORMS": "cpu"},
                ),
                replicas=profile.gpus.count // profile.gpus.per_host,
                coscheduling=(
                    CoschedulingConfig(group_by="nvlink.domain") if profile.gpus.count > profile.gpus.per_host else None
                ),
                priority_band={"interactive": job_pb2.PRIORITY_BAND_INTERACTIVE, "batch": job_pb2.PRIORITY_BAND_BATCH}[
                    profile.priority
                ],
                scheduling_timeout=Duration.from_seconds(plan["series"]["scheduling_timeout"]),
                timeout=Duration.from_seconds(plan["series"]["task_timeout"]),
                max_retries_failure=0,
                max_task_failures=0,
                max_retries_preemption=1,
                existing_job_policy=job_pb2.EXISTING_JOB_POLICY_ERROR,
                task_image=plan["series"]["task_image"],
            )
            write_record(prefix_join(plan["trial_uri"], "plan.json"), plan)
            with track_allocations(client, plan):
                job.wait(timeout=14400, raise_on_failure=False, stream_logs=True)
            status = client.job_status(job.job_id)
            attempts = []
            for summary in client.list_tasks(job.job_id):
                task = client.task_status(summary.task_id)
                for attempt in task.attempts:
                    finish = attempt.finished_at.epoch_ms() / 1000 if attempt.finished_at else None
                    cost, allocation = allocation_cost(
                        plan, attempt.attempt_uid, attempt.pod_uid, finish, profile.gpus.per_host
                    )
                    attempts.append(
                        {
                            "task_id": str(task.task_id),
                            "attempt_uid": attempt.attempt_uid,
                            "state": attempt.state.value,
                            "pod_uid": attempt.pod_uid,
                            "node_name": attempt.node_name,
                            "gpu_seconds": cost,
                            "allocation": allocation,
                            "error": attempt.error_message or attempt.terminal_reason,
                        }
                    )
            result = {"plan": plan, "state": status.state.value, "attempts": attempts, "observed_unix": time.time()}
            write_record(prefix_join(plan["trial_uri"], "execution.json"), result)
            args.output.write_text(json.dumps(result, indent=2))
            print(
                json.dumps(
                    {
                        "state": result["state"],
                        "gpu_hours": sum(row["gpu_seconds"] or 0 for row in attempts) / 3600,
                        "cost_complete": all(row["gpu_seconds"] is not None for row in attempts),
                        "run_uri": plan["trial_uri"],
                    },
                    indent=2,
                )
            )
            raise SystemExit(0 if status.state.value == "succeeded" else 1)


if __name__ == "__main__":
    main()
