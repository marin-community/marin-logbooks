# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the distinct 64-device Gram diagnostic only on ready v4-128 capacity."""

import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def launch_argv(source_commit, variant="v4-128"):
    assert variant == "v4-128"
    devices = 64
    mesh_data, mesh_model = 8, 8
    directory = "logbooks/20-ema-gram-polar"
    script = f"{directory}/scaled_gram_check.py"
    manifest = f"{directory}/scaled-gram-manifest.json"
    prefix = "gs://marin-us-central2/ema-gram-20-scaled64-gram/2026.11.04.01"
    checkpoint = json.loads((ROOT / directory / "loss-pair-checkpoint-metadata.json").read_text())["checkpoint"]
    common = [
        "python",
        script,
        "--source-manifest",
        manifest,
        "--samples",
        "3",
        "--mesh-data",
        str(mesh_data),
        "--mesh-model",
        str(mesh_model),
        "--processes",
        str(devices // 4),
    ]
    logging = [*common, "--mode", "logging-check", "--output-prefix", prefix + "/host-logging"]
    worker = [*common, "--mode", "worker", "--checkpoint", checkpoint, "--output-prefix", prefix]
    command = "JAX_PLATFORMS=cpu " + shlex.join(logging) + " && exec " + shlex.join(worker)
    return [
        "uv",
        "run",
        "--no-sync",
        "iris",
        "--cluster=marin",
        "job",
        "run",
        "--no-wait",
        "--no-cancel-on-exit",
        "--job-name",
        "ema-gram-20-scaled64-gram-v4-128-20261104",
        "--priority",
        "batch",
        "--enable-extra-resources",
        "--tpu",
        variant,
        "--cpu",
        "120",
        "--memory",
        "256GB",
        "--disk",
        "50GB",
        "--region",
        "us-central2",
        "--extra",
        "tpu",
        "--sync-package",
        "marin-core",
        "--sync-package",
        "marin-levanter",
        "--sync-package",
        "marin-dupekit",
        "-e",
        "GIT_COMMIT",
        source_commit,
        "--",
        "bash",
        "-c",
        command,
    ]


def choose_ready_variant(status):
    """Select only 64 JAX devices; the completed 32-device trial is not repeated."""
    ready = {}
    for group in status["groups"]:
        variant = group["device_variant"]
        if group["device_type"] != "tpu" or variant != "v4-128":
            continue
        if group["region"] != "us-central2":
            continue
        for allocation in group.get("slices", []):
            if allocation.get("state") != "ready" or allocation.get("capacity_status") != "available":
                continue
            hosts = 16
            vms = allocation.get("vms", [])
            if len(vms) == hosts and all(v.get("worker_healthy") and v.get("running_task_count") == 0 for v in vms):
                ready.setdefault(variant, []).append(allocation["slice_id"])
    for variant in ("v4-128",):
        if variant in ready:
            return variant, ready[variant]
    return None, []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.source_commit
    base = Path(__file__).parent
    manifest = json.loads((base / "scaled-gram-manifest.json").read_text())
    for path, identity in manifest["files"].items():
        raw = (ROOT / path).read_bytes()
        assert len(raw) == identity["bytes"] and hashlib.sha256(raw).hexdigest() == identity["sha256"]
    for file in [
        "scaled-gram-logging-validation.json",
        "scaled-gram-cpu8-wandb-validation.json",
        "scaled-gram-cpu16-wandb-validation.json",
    ]:
        evidence = json.loads((base / file).read_text())
        assert evidence["state"] == "finished" and evidence["source_files"] == manifest["files"]
    for count in (8, 16):
        cpu = json.loads((base / f"scaled-gram-cpu{count}-validation.json").read_text())
        assert cpu["completed"] and cpu["source_files"] == manifest["files"]
    for mode in ["single", "matrix"]:
        evidence = json.loads((base / f"grouped-gram-regression-{mode}-validation.json").read_text())
        assert evidence["exit_code"] == 0 and evidence["api_verified"]
        assert all(hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == h for f, h in evidence["files"].items())
    scratch = ROOT / "scratch/20261104-scaled64-selection.json"
    if scratch.exists():
        prior = json.loads(scratch.read_text())
        if prior.get("launch_started"):
            raise RuntimeError(
                "A launch was already attempted; preserve its identity and investigate without resubmission"
            )
    if args.launch:
        # Publish the exact launch source before allocating compute.
        remote = subprocess.check_output(
            ["git", "ls-remote", "logbooks", "refs/heads/codex/20-ema-gram-polar"], cwd=ROOT, text=True
        ).split()
        assert remote and remote[0] == args.source_commit, "Publish this exact HEAD before launch"
        assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT)
    # Refresh only fleet capacity; no terminal job or observer is polled.
    raw = subprocess.check_output(
        ["uv", "run", "--no-sync", "iris", "--cluster=marin", "rpc", "controller", "get-autoscaler-status"],
        cwd=ROOT,
        text=True,
    )
    capacity = json.loads(raw)["status"]
    variant, slices = choose_ready_variant(capacity)
    record = dict(
        source_commit=args.source_commit,
        observed_utc=datetime.now(UTC).isoformat(),
        variant=variant,
        ready_slices=slices,
        capacity=capacity,
        launched=False,
    )
    scratch.write_text(json.dumps(record, indent=2) + "\n")
    if variant is None:
        print("No suitable ready v4-128 slice; no job submitted.", flush=True)
        return
    argv = launch_argv(args.source_commit, variant)
    print(json.dumps(dict(variant=variant, ready_slices=slices, argv=argv)), flush=True)
    if args.launch:
        # Capacity is advisory, not a reservation; the scheduler can still queue.
        record["launch_started"] = True
        record["argv"] = argv
        scratch.write_text(json.dumps(record, indent=2) + "\n")
        subprocess.run(argv, cwd=ROOT, check=True)
        record["launched"] = True
        scratch.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
