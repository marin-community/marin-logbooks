# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect and verify only accepted terminal expansion identities; default is local plan."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path

from collect_r02_terminal_tasks import collect_task
from expansion_worker import resolve
from prepare_expansion_terminal_collection import prepare
from verify_expansion_artifacts import verify_artifacts


def finish(
    worktree, registry_path, campaign, bundle_path, completions_path, accepted_path, joins_path, mode, index=None
):
    worktree = worktree.resolve()
    shared = worktree.parent / "ema-gram-research-20/scratch/20261105-subset-expansion-registry.json"
    if registry_path.resolve() != shared.resolve():
        raise ValueError("Use the permanent expansion registry")
    registry = json.loads(registry_path.read_text())
    binding = registry["campaigns"][campaign]
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip() != binding["source"]:
        raise ValueError("Execution checkout differs from accepted source")
    for path in [bundle_path, completions_path, accepted_path]:
        if path.is_absolute() or ".." in path.parts or str(path) not in binding["inputs"]:
            raise ValueError("Input is not bound to accepted campaign")
        if hashlib.sha256((worktree / path).read_bytes()).hexdigest() != binding["inputs"][str(path)]:
            raise ValueError("Frozen campaign input differs")
    bundle = json.loads((worktree / bundle_path).read_text())
    joins = json.loads(joins_path.read_text())["jobs"] if joins_path is not None else {}
    completed = {}
    for number in range(len(bundle["cells"])):
        p = worktree / "scratch" / f"{campaign}-{number:02d}-completion" / "verified.json"
        if p.exists():
            record = json.loads(p.read_text())
            job = f"/kaiyuew@stanford.edu/{campaign}-{number:02d}"
            if record["job_id"] != job:
                raise ValueError("Completion output belongs to another job")
            completed[job] = record
    plan = prepare(worktree, registry, campaign, bundle_path, joins, completed)
    if mode == "plan":
        return plan
    directory = worktree / "scratch" / f"{campaign}-terminal-tasks"
    directory.mkdir(exist_ok=True)
    with (directory / "owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if mode == "collect":
            tasks = [t for request in plan["requests"] for t in request["task_ids"]]

            def execute(argv):
                return subprocess.run(argv, cwd=worktree, capture_output=True, text=True)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda t: collect_task(t, directory, execute), tasks))
            return dict(tasks_collected_or_reused=len(results), scientifically_complete=False)
        if mode != "verify" or index is None:
            raise ValueError("Verification requires one exact expansion index")
        job = f"/kaiyuew@stanford.edu/{campaign}-{index:02d}"
        if job in plan["completed_reused"]:
            return completed[job]
        requests = [r for r in plan["requests"] if r["index"] == index]
        if len(requests) != 1:
            raise ValueError("No accepted canonical terminal request; no online/storage query permitted")
        request = requests[0]
        # Reconstruct the exact configuration before any metric/storage read.
        _, row, _ = resolve(worktree / bundle_path, worktree / completions_path, worktree / accepted_path, index)
        if row["config"] != request["config"] or row["config_sha256"] != request["config_sha256"]:
            raise ValueError("Native reconstruction differs from terminal request")
        out = worktree / "scratch" / f"{campaign}-{index:02d}-completion"
        # Reference alpha0 schema is read locally from the research worktree, never re-polled.
        research = worktree.parent / "ema-gram-research-20"
        return verify_artifacts(research, request, directory, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["worktree", "registry", "bundle", "completions", "accepted-expansions"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--joins", type=Path)
    parser.add_argument("--mode", choices=["plan", "collect", "verify"], default="plan")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            finish(
                args.worktree,
                args.registry,
                args.campaign,
                args.bundle,
                args.completions,
                args.accepted_expansions,
                args.joins,
                args.mode,
                args.index,
            ),
            indent=2,
        )
    )
