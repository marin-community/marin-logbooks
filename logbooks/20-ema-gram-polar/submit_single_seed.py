# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch distinct remaining scientific cells with a persistent per-cell guard."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).parent
LEDGER = ROOT / "scratch/20261104-single-seed-ledger.json"
MUTEX = threading.Lock()


def ready_slots(status):
    slots = []
    for group in status["groups"]:
        variant = group["device_variant"]
        if group["device_type"] != "tpu" or group["region"] != "us-central2":
            continue
        if variant not in ("v4-8", "v4-32"):
            continue
        for allocation in group.get("slices", []):
            vms = allocation.get("vms", [])
            if (
                allocation.get("state") == "ready"
                and allocation.get("capacity_status") == "available"
                and len(vms) == (1 if variant == "v4-8" else 4)
                and all(v.get("worker_healthy") and v.get("running_task_count") == 0 for v in vms)
            ):
                slots.append((variant, allocation["slice_id"]))
    return sorted(slots, key=lambda s: s[0] != "v4-32")


def command(index, variant, source):
    job = f"ema-gram-20-single-seed-{index:02d}-20261104"
    common = ["python", "-m", "experiments.ema_gram.campaign", "--index", str(index), "--variant", variant]
    logging = shlex.join([*common, "--mode", "logging-check", "--output", "/tmp/campaign-host-logging.json"])
    worker = shlex.join([*common, "--mode", "worker", "--output", "/tmp/campaign-worker.json"])
    shell = 'export PYTHONPATH="$PWD/lib/levanter/src:$PWD/lib/marin/src:$PWD"\n'
    shell += f"JAX_PLATFORMS=cpu {logging} && exec {worker}"
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
        job,
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
        source,
        "--",
        "bash",
        "-c",
        shell,
    ]


def persist(ledger):
    temp = LEDGER.with_suffix(".tmp")
    with temp.open("w") as handle:
        handle.write(json.dumps(ledger, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(LEDGER)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--watch", action="store_true", help="Recheck unused ready capacity every 60 seconds")
    parser.add_argument("--allow-cold-queue", action="store_true", help="Requires explicit user authorization")
    args = parser.parse_args()
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.source_commit
    preflight = json.loads((BASE / "single-seed-preflight.json").read_text())
    for path, sha in preflight["source_files"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha, path
    validation = json.loads((BASE / "single-seed-validation.json").read_text())
    assert validation["source_files"] == preflight["source_files"] and validation["passed"]
    cells = json.loads((BASE / "single-seed-pilots.json").read_text())
    if args.launch:
        assert (
            subprocess.check_output(
                ["git", "ls-remote", "logbooks", "refs/heads/codex/20-ema-gram-single-seed-20261104"], text=True
            ).split()[0]
            == args.source_commit
        )
        assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    with LEDGER.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"cells": {}, "observations": []}
        raw = subprocess.check_output(
            ["uv", "run", "--no-sync", "iris", "--cluster=marin", "rpc", "controller", "get-autoscaler-status"],
            text=True,
        )
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        capacity_path = ROOT / f"scratch/{stamp}-single-seed-capacity.json"
        capacity_path.write_text(raw)
        slots = ready_slots(json.loads(raw)["status"])
        remaining = [i for i in range(len(cells)) if str(i) not in ledger["cells"]]
        selected = []
        for variant, allocation in slots:
            # v4-8 first serves cheap alpha-zero cells; larger slots prioritize nonzero cells.
            candidates = sorted(remaining, key=lambda i: (cells[i]["alpha"] == 0) != (variant == "v4-8"))
            if candidates:
                index = candidates[0]
                selected.append((index, variant, allocation))
                remaining.remove(index)
        if args.allow_cold_queue:
            selected.extend((i, "v4-8" if cells[i]["alpha"] == 0 else "v4-32", None) for i in remaining)
        ledger["observations"].append(dict(time=stamp, capacity=str(capacity_path.relative_to(ROOT)), slots=slots))
        persist(ledger)
        print(json.dumps(dict(selected=selected, source=args.source_commit, launch=args.launch)), flush=True)
        if not args.launch:
            return False

        def submit(item):
            index, variant, allocation = item
            argv = command(index, variant, args.source_commit)
            record = dict(
                cell=cells[index],
                variant=variant,
                advisory_slice=allocation,
                source_commit=args.source_commit,
                argv=argv,
                launch_started=True,
                job_id=f"/kaiyuew@stanford.edu/ema-gram-20-single-seed-{index:02d}-20261104",
                submitted_utc=datetime.now(UTC).isoformat(),
                launched=False,
            )
            with MUTEX:
                ledger["cells"][str(index)] = record
                persist(ledger)
            output = ROOT / f"scratch/20261104-single-seed-{index:02d}-submit.log"
            with output.open("w") as handle:
                result = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT)
            with MUTEX:
                record.update(
                    submit_exit_code=result.returncode,
                    launched=result.returncode == 0,
                    submit_log=str(output.relative_to(ROOT)),
                )
                persist(ledger)
            print(json.dumps(dict(index=index, **record)), flush=True)

        if selected:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as pool:
                list(pool.map(submit, selected))
        return len(ledger["cells"]) < len(cells)


if __name__ == "__main__":
    while main():
        if "--watch" not in sys.argv:
            break
        time.sleep(60)
