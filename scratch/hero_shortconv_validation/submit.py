# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare or submit a task-local exact-wheel ShortConv validation."""

import argparse
import ast
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
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
SOURCE_ROOT = Path("/private/tmp/hero-shortconv-cudagraph-39e-20261104")
SOURCE_REVISION = "39e62869693c46402b1a95fde4fc55ca1aab9ae1"
PATCH_DIGEST = "1b377716b4586cc8ad5e4c607048aa6bb39ce159893da847cf6514ba9aae1896"
MANIFEST_FILENAME = "hero-shortconv-manifest.json"
WORKER_FILENAME = "hero-shortconv-worker.py"
TINY_ENGINE_FILENAME = "hero-shortconv-tiny-engine.py"
TEST_SOURCE_FILENAME = "hero-shortconv-test-source.tar.gz"
OVERLAY_FILENAMES = {
    "vllm/config/compilation.py": "hero-shortconv-overlay-compilation.py",
    "vllm/model_executor/models/config.py": "hero-shortconv-overlay-model-config.py",
    "vllm/model_executor/models/grugmoe.py": "hero-shortconv-overlay-grugmoe.py",
}
MANAGED_INPUTS = (
    "vendor/skyrl/pyproject.toml",
    "vendor/skyrl/uv.lock",
    "vendor/skyrl/.uv-version",
    "vendor/skyrl/cloud/iris/bootstrap_runtime.sh",
    "vendor/skyrl/cloud/iris/runtime_environment.py",
    "vendor/skyrl/marinskyrl/environment_contract.py",
)
TEST_INPUTS = (
    "pyproject.toml",
    "tests/__init__.py",
    "tests/ci_envs.py",
    "tests/conftest.py",
    "tests/utils.py",
    "tests/models/__init__.py",
    "tests/models/utils.py",
    "tests/models/registry.py",
    "tests/models/test_grugmoe.py",
    "infra/release",
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
    if git("rev-parse", "HEAD", cwd=SOURCE_ROOT).decode().strip() != SOURCE_REVISION:
        raise ValueError("Test source must remain on exact 39e628")
    patch = MARIN_ROOT / "scratch/hero-shortconv-piecewise-39e-candidate.patch"
    if digest(patch.read_bytes()) != PATCH_DIGEST:
        raise ValueError("Reviewed source patch digest changed")
    source_manifest = json.loads((MARIN_ROOT / "scratch/hero-shortconv-piecewise-39e-manifest.json").read_text())
    if source_manifest["patch_sha256"] != PATCH_DIGEST or source_manifest["base_commit"] != SOURCE_REVISION:
        raise ValueError("Source manifest differs from reviewed candidate")
    for item in source_manifest["files"]:
        if digest((SOURCE_ROOT / item["path"]).read_bytes()) != item["patched_sha256"]:
            raise ValueError(f"Frozen candidate changed: {item['path']}")
        if digest(git("show", f"{SOURCE_REVISION}:{item['path']}", cwd=SOURCE_ROOT)) != item["base_sha256"]:
            raise ValueError(f"Candidate preimage differs from source pin: {item['path']}")
    workspace = destination / "marin"
    destination.mkdir(parents=True, exist_ok=True)
    if not workspace.exists():
        subprocess.run(["git", "worktree", "add", "--detach", str(workspace), revision], cwd=MARIN_ROOT, check=True)
    if git("rev-parse", "HEAD", cwd=workspace).decode().strip() != revision:
        raise ValueError("Prepared workspace is pinned to another Marin revision")
    if git("status", "--porcelain", cwd=workspace).strip():
        raise ValueError("Prepared immutable Marin workspace is dirty")

    archive = git("archive", "--format=tar", SOURCE_REVISION, *TEST_INPUTS, cwd=SOURCE_ROOT)
    output = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(archive)) as source, tarfile.open(fileobj=output, mode="w") as target:
        for member in source:
            if not member.isfile():
                continue
            content = (
                (SOURCE_ROOT / member.name).read_bytes()
                if member.name == "tests/models/test_grugmoe.py"
                else source.extractfile(member).read()
            )
            member.size = len(content)
            member.mtime = 0
            member.uid = member.gid = 0
            member.uname = member.gname = ""
            target.addfile(member, io.BytesIO(content))
    payload = {TEST_SOURCE_FILENAME: gzip.compress(output.getvalue(), mtime=0)}
    for item in source_manifest["files"]:
        if item["runtime_overlay"]:
            payload[OVERLAY_FILENAMES[item["path"]]] = (SOURCE_ROOT / item["path"]).read_bytes()
    for filename, staged_name in (("worker.py", WORKER_FILENAME), ("tiny_engine.py", TINY_ENGINE_FILENAME)):
        payload[staged_name] = Path(__file__).with_name(filename).read_bytes()
    lock = tomllib.loads((workspace / "vendor/skyrl/uv.lock").read_text())
    wheel = next(
        package
        for package in lock["package"]
        if package["name"] == "vllm" and "aarch64" in package.get("source", {}).get("url", "")
    )
    if "39e62869693c" not in wheel["source"]["url"]:
        raise ValueError("Managed ARM wheel changed")
    tools = {
        name: next(package["version"] for package in lock["package"] if package["name"] == name)
        for name in ("pytest", "pluggy", "iniconfig", "pygments")
    }
    tools["tblib"] = "3.2.2"
    manifest = {
        "marin_revision": revision,
        "source_revision": SOURCE_REVISION,
        "patch_sha256": PATCH_DIGEST,
        "wheel": {"version": wheel["version"], **wheel["wheels"][0]},
        "managed_inputs": {path: digest((workspace / path).read_bytes()) for path in MANAGED_INPUTS},
        "overlay": [
            {**item, "payload_path": OVERLAY_FILENAMES[item["path"]]}
            for item in source_manifest["files"]
            if item["runtime_overlay"]
        ],
        "test_source_payload": TEST_SOURCE_FILENAME,
        "tiny_engine_payload": TINY_ENGINE_FILENAME,
        "payload": {path: digest(content) for path, content in payload.items()},
        "test_tools": tools,
        "task_image": task_image(workspace),
        "expected_core": {"torch": "2.13.0", "torchvision": "0.28.0", "transformers": "5.16.1"},
    }
    payload[MANIFEST_FILENAME] = (json.dumps(manifest, indent=2) + "\n").encode()
    for path, content in payload.items():
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return workspace, payload, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-dir", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--gpus", type=int, choices=(1, 4), default=1)
    parser.add_argument("--stage", choices=("focused", "engine", "both"), default="both")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    workspace, payload, manifest = prepare(args.prepare_dir.resolve())
    resources = ResourceConfig(
        cpu=32,
        ram="64GiB",
        disk="128GiB",
        device=GpuConfig(variant="GB200", count=args.gpus),
        target_cluster="cw-us-east-08a",
    )
    command = [
        "python3.12",
        WORKER_FILENAME,
        "--manifest",
        MANIFEST_FILENAME,
        "--manifest-sha256",
        digest(payload[MANIFEST_FILENAME]),
        "--stage",
        args.stage,
    ]
    record = {
        "job_id": f"/dlwh/{args.job_name}",
        "status": "prepared",
        "workspace": str(workspace),
        "manifest_sha256": digest(payload[MANIFEST_FILENAME]),
        "source_revision": SOURCE_REVISION,
        "marin_revision": manifest["marin_revision"],
        "wheel": manifest["wheel"],
        "patch_sha256": PATCH_DIGEST,
        "command": command,
        "resources": {"gpu": args.gpus, "cpu": 32, "memory": "64GiB", "disk": "128GiB"},
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
                    env_vars={"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_V2_MODEL_RUNNER": "1"},
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
