# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit validated seed0 confirmations once, with a separate permanent campaign ledger."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

from prepare_subset_confirmations import prepare
from submit_subset_training import command as pilot_command
from terminal_worker_identity import completed_workers

from experiments.ema_gram.plan import Cell, read_lock

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).parent


def validate_completions(lock, records):
    rows = {Cell(**r["cell"]).name: r for r in records["results"]}
    if len(rows) != len(records["results"]) or set(rows) != {r.name for r in lock.results}:
        raise ValueError("Completion evidence must cover exactly the locked pilot and expansion cells")
    for result in lock.results:
        r = rows[result.name]
        if result.status != "completed":
            raise ValueError("Scientific failure adjudication requires a separate reviewed record before launch")
        if not all(r[k] for k in ("scientifically_complete", "metric_history_complete", "exact_subset_config_verified")):
            raise ValueError("Pilot evidence is not scientifically complete")
        if r["remaining_gates"] or r["online_state"] != "finished":
            raise ValueError("Pilot validation or online tracking is unfinished")
        if (r["stored_native_counter"], r["expected_steps"], r["native_final_label"]) != (7630, 7630, 7629):
            raise ValueError("Pilot did not finish its native schedule")
        if r["final_c4_loss"] != result.validation_loss or r["run_url"] != result.run_url:
            raise ValueError("Locked result differs from verified completion")
        if r["canonical_job"]["state"] != "JOB_STATE_SUCCEEDED" or r["canonical_job"]["exit_code"] != 0:
            raise ValueError("Canonical pilot outcome is not success")
        tasks = r["tasks"] if "tasks" in r else [r["task"]]
        expected_tasks = 1 if r["cell"]["alpha"] == 0 else 4
        if r["canonical_job"]["job_id"] != r["job_id"]:
            raise ValueError("Canonical job identity differs from completion record")
        if {t["task_id"] for t in tasks} != {f"{r['job_id']}/{i}" for i in range(expected_tasks)}:
            raise ValueError("Task identities do not belong to the completed job")
        if len(tasks) != expected_tasks or len(set(completed_workers(r, tasks))) != expected_tasks:
            raise ValueError("Incomplete task evidence")
        for task in tasks:
            attempts = task["attempts"]
            if not attempts or task["state"] != "TASK_STATE_SUCCEEDED" or task["exit_code"] != 0:
                raise ValueError("Missing successful task attempt")
            current = [a for a in attempts if a["attempt_id"] == task["current_attempt_id"]]
            if len(current) != 1 or current[0]["state"] != "TASK_STATE_SUCCEEDED" or current[0]["exit_code"] != 0:
                raise ValueError("Current task attempt did not succeed")
            if len({a["attempt_uid"] for a in attempts}) != len(attempts):
                raise ValueError("Repeated task attempt identity")
        if not r["run_attempt_join_evidence"] or r["checkpoint"]["arrays"] != 57:
            raise ValueError("Missing native run join or checkpoint manifest")
        if len({h["id"] for h in r["host_logging"]}) < expected_tasks or any(
            h["state"] != "finished" or not h["source_files_verified"] for h in r["host_logging"]
        ):
            raise ValueError("Missing online execution-host logging evidence")


def command(index, variant, source, campaign, bundle, lock):
    argv = pilot_command(index, variant, source)
    argv[argv.index("--job-name") + 1] = f"{campaign}-{index:02d}"
    common = [
        "python",
        "logbooks/20-ema-gram-polar/confirmation_worker.py",
        "--bundle",
        str(bundle),
        "--lock",
        str(lock),
        "--index",
        str(index),
    ]
    logging = shlex.join([*common, "--mode", "logging-check", "--output", "/tmp/confirmation-host-logging.json"])
    worker = shlex.join([*common, "--mode", "worker", "--output", "/tmp/confirmation-worker.json"])
    gate = (
        "python logbooks/20-ema-gram-polar/check_campaign_caches.py "
        "--manifest logbooks/20-ema-gram-polar/subset-cache-manifest.json "
        "--output /tmp/confirmation-cache-check.json"
    )
    argv[-1] = (
        'export PYTHONPATH="$PWD/logbooks/20-ema-gram-polar:$PWD/lib/levanter/src:$PWD/lib/marin/src:$PWD"\n'
        + f"{gate} && JAX_PLATFORMS=cpu {logging} && exec {worker}"
    )
    return argv


def persist(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("w") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def claim(ledger, index, record):
    key = str(index)
    if key in ledger["cells"]:
        raise ValueError("Cell was already guarded; reconcile its existing identity instead of resubmitting")
    ledger["cells"][key] = dict(record, launch_started=True, launched=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--completions", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--published-ref", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"ema-gram-20-confirm-seed0-[a-z0-9-]+", args.campaign):
        raise ValueError("Use a distinct confirmation campaign identity")
    paths = [args.bundle, args.lock, args.completions]
    for path in paths:
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Inputs must be tracked paths within the immutable execution worktree")
        subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], check=True, stdout=subprocess.DEVNULL)
    lock = read_lock(args.lock)
    bundle = json.loads(args.bundle.read_text())
    if bundle != prepare(lock, bundle["artifact_version"]):
        raise ValueError("Frozen bundle does not match locked configurations")
    validate_completions(lock, json.loads(args.completions.read_text()))
    qualification = json.loads((BASE / "confirmation-worker-preflight.json").read_text())
    if not qualification["passed"] or not qualification["logging_check"]["success"]:
        raise ValueError("Confirmation adapter qualification is incomplete")
    if qualification["source_sha256"] != hashlib.sha256((BASE / "confirmation_worker.py").read_bytes()).hexdigest():
        raise ValueError("Qualified worker source changed")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != args.source_commit:
        raise ValueError("Execution HEAD differs from requested source")
    commands = [
        command(i, row["variant"], args.source_commit, args.campaign, args.bundle, args.lock)
        for i, row in enumerate(bundle["cells"])
    ]
    if not args.launch:
        print(json.dumps(dict(submitted=False, commands=commands)))
        return
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("Publish tracked changes before submission")
    if (
        subprocess.check_output(["git", "ls-remote", "logbooks", args.published_ref], text=True).split()[0]
        != args.source_commit
    ):
        raise ValueError("Exact source has not been published on the supplied ref")
    ledger_path = ROOT / "scratch" / (args.campaign + "-ledger.json")
    ledger_path.parent.mkdir(exist_ok=True)
    with ledger_path.with_suffix(".lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = dict(
            source=args.source_commit, inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        )
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(binding=binding, cells={})
        if ledger["binding"] != binding:
            raise ValueError("Campaign ledger is bound to a different source or input set")
        remaining = [i for i in range(len(commands)) if str(i) not in ledger["cells"]]
        if not remaining:
            print("All confirmation cells already guarded; no submission attempted.")
            return
        subprocess.run(
            [
                sys.executable,
                str(BASE / "check_campaign_caches.py"),
                "--full",
                "--manifest",
                str(BASE / "subset-cache-manifest.json"),
                "--output",
                str(ledger_path.with_suffix(".cache-check.json")),
            ],
            check=True,
        )
        mutex = threading.Lock()

        def submit(index):
            record = dict(
                job_id=f"/kaiyuew@stanford.edu/{args.campaign}-{index:02d}",
                argv=commands[index],
                submitted_utc=datetime.now(UTC).isoformat(),
            )
            with mutex:
                claim(ledger, index, record)
                persist(ledger_path, ledger)
            output = ledger_path.parent / f"{args.campaign}-{index:02d}-submit.log"
            with output.open("w") as stream:
                result = subprocess.run(commands[index], stdout=stream, stderr=subprocess.STDOUT)
            with mutex:
                ledger["cells"][str(index)].update(
                    submit_exit_code=result.returncode, launched=result.returncode == 0, submit_log=str(output)
                )
                persist(ledger_path, ledger)
            return result.returncode

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(remaining)) as pool:
            codes = list(pool.map(submit, remaining))
        if any(codes):
            raise RuntimeError("Submission error: preserve guards and reconcile exact canonical identities")


if __name__ == "__main__":
    main()
