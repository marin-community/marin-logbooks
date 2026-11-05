# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One exclusive observer per immutable expansion campaign; no submissions."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from submit_subset_expansions import persist_registry

from experiments.ema_gram.plan import Cell

TERMINAL = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_KILLED",
    "JOB_STATE_WORKER_FAILED",
    "JOB_STATE_UNSCHEDULABLE",
}


def accepted_entries(registry, campaign, source, bundle):
    binding = registry["campaigns"][campaign]
    if binding["source"] != source or binding["artifact_version"] != bundle["artifact_version"]:
        raise ValueError("Campaign source/version mismatch")
    accepted = {}
    for name, row in registry["cells"].items():
        if row["campaign"] != campaign:
            continue
        index = row["index"]
        if not isinstance(index, int) or not 0 <= index < len(bundle["cells"]):
            raise ValueError("Unexpected campaign index")
        expected = bundle["cells"][index]
        if (
            name != Cell(**expected["cell"]).name
            or row["cell"] != expected["cell"]
            or row["source_commit"] != source
            or row["config_sha256"] != expected["config_sha256"]
            or row["job_id"] != f"/kaiyuew@stanford.edu/{campaign}-{index:02d}"
            or not row["launch_started"]
        ):
            raise ValueError("Expansion identity differs from frozen bundle")
        if row.get("launched") and row.get("submit_exit_code") == 0:
            accepted[str(index)] = row
    return accepted


def register(state, index, entry, now):
    identity = {key: entry[key] for key in ("job_id", "source_commit", "config_sha256")}
    if index in state:
        if any(state[index][key] != value for key, value in identity.items()):
            raise ValueError("Existing observer identity changed")
        return False
    state[index] = dict(
        identity, first_seen=now, next_check=now + 120, terminal=False, follow_started=True, owner_pid=os.getpid()
    )
    return True


def apply_check(record, result, now):
    if "job" in result and result["job"]["job_id"] != record["job_id"]:
        raise ValueError("Canonical response belongs to a different job")
    record["latest"] = result
    record["next_check"] = now + 570
    if "job" in result:
        record["last_successful_check"] = now
        record["last_successful_snapshot"] = result
        record["terminal"] = result["job"]["state"] in TERMINAL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worktree", "registry", "bundle"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch("ema-gram-20-expand-seed0-[a-z0-9-]+", args.campaign):
        raise ValueError("Invalid campaign")
    worktree = args.worktree.resolve()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip()
    if source != args.source_commit:
        raise ValueError("Execution worktree source changed")
    if args.bundle.is_absolute() or ".." in args.bundle.parts:
        raise ValueError("Bundle must be a tracked relative path")
    registry_path = worktree.parent / "ema-gram-research-20/scratch/20261105-subset-expansion-registry.json"
    if args.registry.resolve() != registry_path.resolve():
        raise ValueError("Use the shared permanent registry")
    registry = json.loads(registry_path.read_text())
    binding = registry["campaigns"][args.campaign]
    for name, sha in binding["inputs"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or hashlib.sha256((worktree / path).read_bytes()).hexdigest() != sha:
            raise ValueError("Frozen campaign input changed")
    if str(args.bundle) not in binding["inputs"]:
        raise ValueError("Bundle is not bound to campaign")
    bundle = json.loads((worktree / args.bundle).read_text())
    accepted_entries(registry, args.campaign, source, bundle)
    scratch = worktree / "scratch"
    scratch.mkdir(exist_ok=True)
    state_path = scratch / f"{args.campaign}-observers.json"
    marker = state_path.with_suffix(".initialized.json")
    base = ["uv", "run", "--no-sync", "iris", "--cluster=marin"]
    with state_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if marker.exists() and not state_path.exists():
            raise ValueError("Observer state missing: reconcile existing follow identities before recovery")
        if state_path.exists():
            saved = json.loads(state_path.read_text())
            if not marker.exists() or saved["binding"] != binding:
                raise ValueError("Observer binding/initialization mismatch")
        else:
            if list(scratch.glob(f"{args.campaign}-*-follow.log")):
                raise ValueError("Existing follower evidence requires reconciliation")
            persist_registry(marker, dict(binding=binding))
            saved = dict(binding=binding, jobs={})
            persist_registry(state_path, saved)
        state = saved["jobs"]
        followers, futures = {}, {}

        def save():
            saved["owner_pid"] = os.getpid()
            persist_registry(state_path, saved)

        def check(index, job_id):
            path = scratch / f"{args.campaign}-{index}-canonical-{time.time_ns()}.json"
            with path.open("x") as handle:
                result = subprocess.run(
                    [*base, "rpc", "controller", "get-job-status", "--job-id", job_id],
                    cwd=worktree,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode:
                return dict(error=f"Canonical RPC exit {result.returncode}", path=str(path))
            return dict(job=json.loads(path.read_text())["job"], path=str(path))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            while True:
                registry = json.loads(registry_path.read_text())
                if registry["campaigns"][args.campaign] != binding:
                    raise ValueError("Campaign binding changed while observing")
                accepted = accepted_entries(registry, args.campaign, source, bundle)
                if set(state) - set(accepted):
                    raise ValueError("Previously accepted identities disappeared; preserve state")
                for index, entry in accepted.items():
                    if register(state, index, entry, time.time()):
                        save()  # Permanent follow guard precedes Popen; never retry an ambiguous follow.
                        log = scratch / f"{args.campaign}-{index}-follow.log"
                        with log.open("x") as handle:
                            follow = subprocess.Popen(
                                [*base, "job", "logs", entry["job_id"], "--follow"],
                                cwd=worktree,
                                stdout=handle,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                        followers[index] = follow
                        state[index].update(follow_pid=follow.pid, follow_log=str(log))
                        save()
                for index, future in list(futures.items()):
                    if future.done():
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = dict(error=repr(exc))
                        apply_check(state[index], result, time.time())
                        del futures[index]
                        save()
                for index, record in state.items():
                    if not record["terminal"] and index not in futures and time.time() >= record["next_check"]:
                        futures[index] = pool.submit(check, index, record["job_id"])
                    if index in followers and followers[index].poll() is not None:
                        record["follow_exit_code"] = followers.pop(index).returncode
                        save()
                if len(state) == len(bundle["cells"]) and all(r["terminal"] for r in state.values()) and not followers:
                    return  # Canonical termination alone is not scientific completion.
                time.sleep(5)


if __name__ == "__main__":
    main()
