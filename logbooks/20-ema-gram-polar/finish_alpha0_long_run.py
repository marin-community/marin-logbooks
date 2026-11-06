# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect and verify the accepted early baseline using its original frozen inputs."""
import argparse
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path

from alpha0_long_run import accepted_entries, command, committed_binding, resolve, validate_protocol
from collect_r02_terminal_tasks import collect_task
from verify_expansion_artifacts import cached_json, verify_artifacts


def prepare(root, receipt_path, startup_path):
    receipt = json.loads(receipt_path.read_text())
    worktree = Path(receipt["execution_worktree"])
    source, campaign = receipt["source"], receipt["campaign"]
    paths = {k: Path(v) for k, v in receipt["paths"].items()}
    binding = committed_binding(worktree, source, list(paths.values()))
    _, bundle = resolve(worktree / paths["bundle"], worktree / paths["completions"])
    protocol = json.loads((worktree / paths["protocol"]).read_text())
    if validate_protocol(protocol, bundle) != receipt["protocol_digest"]:
        raise ValueError("Accepted protocol digest changed")
    ledger = json.loads((worktree / f"scratch/{campaign}-ledger.json").read_text())
    if ledger != receipt["ledger"]:
        raise ValueError("Accepted baseline ledger changed")
    argv = command(source, campaign, paths["bundle"], paths["completions"], paths["protocol"])
    entries = accepted_entries(ledger, binding, campaign, argv, bundle)
    if entries != receipt["accepted"] or set(entries) != {"0"}:
        raise ValueError("Baseline receipt acceptance differs")
    observed = json.loads(Path(receipt["observer_state"]).read_text())
    if observed["binding"] != binding or observed["owner_pid"] != receipt["observer_pid"]:
        raise ValueError("Original observer binding differs")
    entry = entries["0"]
    obs = observed["jobs"]["0"]
    if any(obs[k] != v for k, v in entry.items()) or not obs["terminal"]:
        raise ValueError("Original observer has not reconciled accepted terminal job")
    latest = obs["latest"]
    raw = Path(latest["path"]).read_bytes()
    job = latest["job"]
    if json.loads(raw)["job"] != job or job["job_id"] != entry["job_id"]:
        raise ValueError("Canonical job evidence differs")
    if job["state"] != "JOB_STATE_SUCCEEDED" or job["exit_code"] or job["failure_count"] or job["preemption_count"]:
        raise ValueError("Unsuccessful or retried baseline needs lineage adjudication")
    if job["task_count"] != 1 or job["completed_count"] != 1:
        raise ValueError("Baseline terminal topology differs")
    startup = json.loads(startup_path.read_text())
    if not startup["passed"] or startup["source_commit"] != source:
        raise ValueError("Missing validated baseline startup")
    row = bundle["cells"][0]
    worker = Path("logbooks/20-ema-gram-polar/alpha0_long_run.py")
    worker_bytes = subprocess.check_output(["git", "show", f"{source}:{worker}"], cwd=worktree)
    if worker_bytes != (worktree / worker).read_bytes():
        raise ValueError("Frozen worker changed")
    input_evidence = {
        str(p): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
        for p in [receipt_path, startup_path, *(worktree / p for p in paths.values())]
    }
    return worktree, dict(
        index=0,
        campaign=campaign,
        source_commit=source,
        job_id=entry["job_id"],
        canonical_state=job["state"],
        canonical_job=job,
        canonical_evidence=latest["path"],
        canonical_sha256=hashlib.sha256(raw).hexdigest(),
        failure_count=0,
        preemption_count=0,
        task_ids=[entry["job_id"] + "/0"],
        cached_startup_join=startup["jobs"][entry["job_id"]],
        config=row["config"],
        config_sha256=row["config_sha256"],
        cell=row["cell"],
        input_binding=binding,
        bundle_sha256=binding["inputs"][str(paths["bundle"])],
        worker_sha256=hashlib.sha256(worker_bytes).hexdigest(),
        protocol=protocol,
        input_evidence=input_evidence,
        scientifically_complete=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("plan", "collect", "verify"))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--startup", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    worktree, request = prepare(root, args.receipt.resolve(), args.startup.resolve())
    out = worktree / f"scratch/{request['campaign']}-00-completion"
    tasks = worktree / f"scratch/{request['campaign']}-terminal-tasks"
    out.mkdir(exist_ok=True)
    tasks.mkdir(exist_ok=True)
    with (tasks / "owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if cached_json(out / "request.json", lambda: request) != request:
            raise ValueError("Terminal request changed")
        if args.mode == "plan":
            print(json.dumps({"request": str(out / "request.json"), "task_ids": request["task_ids"]}))
            return
        if args.mode == "collect":
            for task in request["task_ids"]:
                collect_task(
                    task, tasks, lambda argv: subprocess.run(argv, cwd=worktree, capture_output=True, text=True)
                )
            print(json.dumps({"terminal_tasks_collected_or_reused": 1}))
            return
        hosts = {h["id"]: h for h in json.loads((args.startup.parent / "online.json").read_text())}
        for task in request["cached_startup_join"]["tasks"]:
            h = hosts[task["host_logging_id"]]
            if h["state"] != "finished":
                raise ValueError("Startup host logging was not finished")
            if cached_json(out / (h["id"] + ".json"), lambda h=h: h) != h:
                raise ValueError("Cached host snapshot changed")
        result = verify_artifacts(root, request, tasks, out, expected_steps=37335)
        print(json.dumps({"verified": str(out / "verified.json"), "final_c4_loss": result["final_c4_loss"]}))


if __name__ == "__main__":
    main()
