# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare or submit the task-local exact-Torch save_on_cpu allocator probe."""

import argparse
import ast
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path

from fray.iris_backend import convert_constraints, convert_resources
from fray.types import GpuConfig, ResourceConfig
from iris.cli.connect import open_iris_client
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from rigging.secrets import SecretBehavior, require_secret
from rigging.timing import Duration

MARIN_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("/private/tmp/hero-piecewise-runtime-20261104")
SOURCE_REVISION = "3c4466634e4b657ea1219e021fab55cfbb5550e2"
OPTIMIZER_SOURCE = "vendor/skyrl/skyrl-train/skyrl_train/distributed/megatron/grug_muonh.py"
MANIFEST_FILENAME = "hero-save-on-cpu-manifest.json"
WORKER_FILENAME = "hero-save-on-cpu-worker.py"
PROBE_FILENAME = "hero-save-on-cpu-probe.py"
OPTIMIZER_FILENAME = "hero-save-on-cpu-muonh.py"
MANAGED_INPUTS = (
    "vendor/skyrl/pyproject.toml",
    "vendor/skyrl/uv.lock",
    "vendor/skyrl/.uv-version",
    "vendor/skyrl/cloud/iris/bootstrap_runtime.sh",
    "vendor/skyrl/cloud/iris/runtime_environment.py",
    "vendor/skyrl/marinskyrl/environment_contract.py",
)


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def git(*args: str, cwd: Path = MARIN_ROOT) -> bytes:
    return subprocess.check_output(["git", *args], cwd=cwd)


def task_image(workspace: Path) -> str:
    tree = ast.parse((workspace / MANAGED_INPUTS[4]).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "IRIS_TASK_IMAGE" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Managed runtime did not declare an immutable task image")


def prepare(destination: Path) -> tuple[Path, dict[str, bytes], dict]:
    revision = git("rev-parse", "HEAD").decode().strip()
    for path in MANAGED_INPUTS:
        if (MARIN_ROOT / path).read_bytes() != git("show", f"{revision}:{path}"):
            raise ValueError(f"Managed dependency input is dirty: {path}")
    workspace = destination / "marin"
    destination.mkdir(parents=True, exist_ok=True)
    if not workspace.exists():
        subprocess.run(["git", "worktree", "add", "--detach", str(workspace), revision], cwd=MARIN_ROOT, check=True)
    if git("rev-parse", "HEAD", cwd=workspace).decode().strip() != revision:
        raise ValueError("Prepared workspace is pinned to another Marin revision")
    if git("status", "--porcelain", cwd=workspace).strip():
        raise ValueError("Prepared immutable Marin workspace is dirty")
    payload = {
        WORKER_FILENAME: Path(__file__).with_name("worker.py").read_bytes(),
        PROBE_FILENAME: Path(__file__).with_name("probe.py").read_bytes(),
        OPTIMIZER_FILENAME: git("show", f"{SOURCE_REVISION}:{OPTIMIZER_SOURCE}", cwd=SOURCE_ROOT),
    }
    lock = tomllib.loads((workspace / "vendor/skyrl/uv.lock").read_text())
    torch_package = next(
        item for item in lock["package"] if item["name"] == "torch" and item["version"] == "2.13.0+cu132"
    )
    (torch_wheel,) = [
        item for item in torch_package["wheels"] if "cp312-cp312" in item["url"] and "aarch64" in item["url"]
    ]
    manifest = {
        "torch_locked_wheel": torch_wheel,
        "marin_revision": revision,
        "source_revision": SOURCE_REVISION,
        "optimizer_source_path": OPTIMIZER_SOURCE,
        "optimizer_payload": OPTIMIZER_FILENAME,
        "optimizer_sha256": digest(payload[OPTIMIZER_FILENAME]),
        "probe_payload": PROBE_FILENAME,
        "managed_inputs": {path: digest((workspace / path).read_bytes()) for path in MANAGED_INPUTS},
        "payload": {path: digest(content) for path, content in payload.items()},
        "task_image": task_image(workspace),
        "expected_core": {"torch": "2.13.0+cu132", "torchvision": "0.28.0+cu132", "transformers": "5.16.1"},
        "scope": "One GB200 heterogeneous save_on_cpu allocator probe, no model loading or source overlay",
    }
    payload[MANIFEST_FILENAME] = (json.dumps(manifest, indent=2) + "\n").encode()
    for path, content in payload.items():
        (destination / path).write_bytes(content)
    return workspace, payload, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-dir", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    workspace, payload, manifest = prepare(args.prepare_dir.resolve())
    resources = ResourceConfig(
        cpu=32,
        ram="64GiB",
        disk="128GiB",
        device=GpuConfig(variant="GB200", count=1),
        target_cluster="cw-us-east-08a",
    )
    command = [
        "python3.12",
        WORKER_FILENAME,
        "--manifest",
        MANIFEST_FILENAME,
        "--manifest-sha256",
        digest(payload[MANIFEST_FILENAME]),
    ]
    record = {
        "job_id": f"/dlwh/{args.job_name}",
        "status": "prepared",
        "workspace": str(workspace),
        "manifest_sha256": digest(payload[MANIFEST_FILENAME]),
        "source_revision": SOURCE_REVISION,
        "marin_revision": manifest["marin_revision"],
        "optimizer_source_sha256": manifest["optimizer_sha256"],
        "command": command,
        "resources": {"gpu": 1, "cpu": 32, "memory": "64GiB", "disk": "128GiB"},
        "target_cluster": "cw-us-east-08a",
        "source_cluster": "marin",
        "priority": "interactive",
        "timeout_minutes": 60,
        "scheduling_timeout_minutes": 60,
        "failure_retries": 0,
        "preemption_retries": 0,
        "task_image": manifest["task_image"],
        "payload_bytes": sum(map(len, payload.values())),
        "payload_files": sorted(payload),
    }
    if args.run:
        with open_iris_client(cluster_name="marin", workspace=workspace, timeout_ms=120000) as client:
            job = client.submit(
                Entrypoint(command=command, workdir_files=payload),
                name=args.job_name,
                user="dlwh",
                resources=convert_resources(resources),
                task_image=manifest["task_image"],
                environment=EnvironmentSpec(
                    setup_scripts=[],
                    env_vars={},
                    secrets={name: require_secret(name) for name in ("CW_KEY_ID", "CW_KEY_SECRET")},
                    secret_behavior=SecretBehavior.ISOLATE,
                ),
                constraints=convert_constraints(resources),
                scheduling_timeout=Duration.from_hours(1),
                timeout=Duration.from_hours(1),
                max_retries_failure=0,
                max_retries_preemption=0,
                max_task_failures=0,
                existing_job_policy=job_pb2.EXISTING_JOB_POLICY_ERROR,
                priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
            )
            record.update(job_id=str(job.job_id), status="submitted")
    (args.prepare_dir / "submission.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
