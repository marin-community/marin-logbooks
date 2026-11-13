"""Delete each pilot task's Daytona snapshot once its trial has finished.

Only names in the precomputed task -> snapshot map (the 200 pilot tasks) are ever deleted.
A task's snapshot is deleted when its trial-level result.json exists in the run's results
root, or during the final sweep after the Harbor job finishes or the stop file appears.

A result counts only if its finished_at is at or after the cutoff, so results left by an
earlier attempt on the same results root do not mark a task that is being retried.

Usage: snapshot_cleaner.py <task-snapshot-map.json> <results root s3 url> <stop file> <log file> <cutoff ISO UTC>
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from daytona import Daytona, DaytonaConfig

WORKTREE = "/Users/dml/src/marin/marin/.claude/worktrees/open-swe-outcome-pilot"
POLL = 30
MAX_SECONDS = 20 * 3600
DELETABLE = ("ACTIVE", "ERROR", "BUILD_FAILED", "INACTIVE")

task_to_snapshot: dict[str, str] = json.loads(Path(sys.argv[1]).read_text())
results_root = sys.argv[2].rstrip("/")
stop_file = Path(sys.argv[3])
log_file = Path(sys.argv[4])
cutoff = sys.argv[5]
client = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_EVAL_API_KEY"]))


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}"
    print(line, flush=True)
    with log_file.open("a") as handle:
        handle.write(line + "\n")


def fsutil(*args: str) -> str:
    result = subprocess.run(["uv", "run", "fsutil", *args], cwd=WORKTREE, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else ""


def result_paths() -> dict[str, list[str]]:
    listing = fsutil("find", f"{results_root}/harbor_jobs/*/*/result.json")
    paths: dict[str, list[str]] = {}
    for line in listing.splitlines():
        parts = line.strip().split("/")
        if len(parts) >= 2 and "__" in parts[-2]:
            paths.setdefault(parts[-2].rsplit("__", 1)[0], []).append(line.strip())
    return paths


def finished_since_cutoff(paths: list[str]) -> bool:
    """True when any of the task's trial results finished at or after the cutoff."""
    for path in paths:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "result.json"
            fsutil("cp", path, str(target))
            if not target.exists():
                continue
            try:
                finished_at = json.loads(target.read_text()).get("finished_at") or ""
            except json.JSONDecodeError:
                continue  # mid-write; look again on the next poll
        if finished_at[:19] >= cutoff[:19]:
            return True
    return False


def job_status() -> tuple[bool, str]:
    """Whether a Harbor job on this results root finished at or after the cutoff, plus its stats."""
    listing = fsutil("find", f"{results_root}/harbor_jobs/*/result.json")
    paths = [line.strip() for line in listing.splitlines() if line.strip()]
    if not paths:
        return False, "no job result yet"
    done = False
    summary = "job result unreadable"
    for path in paths:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "result.json"
            fsutil("cp", path, str(target))
            if not target.exists():
                continue
            try:
                document = json.loads(target.read_text())
            except json.JSONDecodeError:
                continue  # the driver rewrites this file in place; a read can land mid-write
        stats = document.get("stats", {})
        summary = " ".join(
            f"{k[2:]}={stats.get(k)}" for k in ("n_completed_trials", "n_errored_trials", "n_running_trials")
        )
        # A resumed results root still holds the previous attempt's finished job result.
        done = done or (document.get("finished_at") or "")[:19] >= cutoff[:19]
    return done, summary


def live_snapshots() -> dict[str, str]:
    rows = []
    page = 1
    while True:
        response = client.snapshot.list(page=page, limit=100)
        rows.extend(response.items)
        if page >= response.total_pages:
            break
        page += 1
    mine = set(task_to_snapshot.values())
    return {s.name: str(s.state).split(".")[-1] for s in rows if s.name in mine}


def delete(names: list[str]) -> int:
    deleted = 0
    for name in names:
        try:
            client.snapshot.delete(client.snapshot.get(name))
            deleted += 1
        except Exception as error:  # keep the loop alive; report the failure
            log(f"delete failed {name}: {type(error).__name__}: {str(error)[:160]}")
    return deleted


snapshot_to_task = {name: task for task, name in task_to_snapshot.items()}
total_deleted = 0
start = time.time()
log(f"cleaner start: {len(task_to_snapshot)} tasks, cutoff {cutoff}, results {results_root}")
while True:
    # The loop must outlive transient storage and Daytona API failures: a dead cleaner fills
    # the account's snapshot quota within minutes. Log the failure and try again.
    try:
        done, summary = job_status()
        paths = result_paths()
        live = live_snapshots()
        finished = {
            snapshot_to_task[name]
            for name in live
            if finished_since_cutoff(paths.get(snapshot_to_task[name], []))
        }
    except Exception as error:
        log(f"poll failed, retrying: {type(error).__name__}: {str(error)[:200]}")
        time.sleep(POLL)
        continue
    stopping = done or stop_file.exists() or time.time() - start > MAX_SECONDS
    targets = [
        name
        for name, state in live.items()
        if state in DELETABLE and (stopping or snapshot_to_task[name] in finished)
    ]
    total_deleted += delete(targets)
    log(f"finished_live={len(finished)} live={len(live)} deleted_now={len(targets)} deleted_total={total_deleted} job[{summary}]")
    if stopping:
        time.sleep(20)
        remaining = live_snapshots()
        total_deleted += delete([name for name, state in remaining.items() if state in DELETABLE])
        time.sleep(20)
        log(f"cleaner exit: deleted_total={total_deleted} remaining={len(live_snapshots())}")
        break
    time.sleep(POLL)
