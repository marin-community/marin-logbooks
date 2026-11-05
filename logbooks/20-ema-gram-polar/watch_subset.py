# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Own canonical checks and one log follower for each newly accepted campaign job."""

import argparse
import concurrent.futures
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--stage", default="hq01")
    parser.add_argument("--mode", choices=("probe", "build", "assemble", "data-check"), required=True)
    parser.add_argument("--expected", type=int, help="Expected distinct tasks in this stage")
    parser.add_argument("--interval", type=int, default=570)
    parser.add_argument(
        "--adopt-followers", action="store_true", help="Resume ownership without creating replacement follows"
    )
    args = parser.parse_args()
    assert args.interval >= 30
    scratch = args.worktree / "scratch"
    prefix = f"subset-{args.stage}-{args.mode}"
    expected = args.expected if args.expected is not None else (58 if args.mode == "build" else 1)
    state_path = scratch / f"20261104-{prefix}-observers.json"
    base = ["uv", "run", "--no-sync", "iris", "--cluster=marin"]
    with (scratch / f"20261104-{prefix}-observers.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(state_path.read_text()) if state_path.exists() else {}

        def save():
            temp = state_path.with_suffix(".tmp")
            temp.write_text(json.dumps(state, indent=2) + "\n")
            temp.replace(state_path)

        def check(index, job_id):
            path = scratch / f"{prefix}-{index}-canonical-{time.time_ns()}.json"
            with path.open("w") as handle:
                result = subprocess.run(
                    [*base, "rpc", "controller", "get-job-status", "--job-id", job_id],
                    cwd=args.worktree,
                    stdout=handle,
                )
            if result.returncode:
                return {"error": f"Canonical RPC exit {result.returncode}", "path": str(path)}
            return {"job": json.loads(path.read_text())["job"], "path": str(path)}

        futures = {}
        followers = {}
        adopted = set()
        if args.adopt_followers:
            for index, record in state.items():
                record.setdefault("observer_handoffs", []).append(dict(pid=os.getpid(), time=time.time()))
                if not record["terminal"]:
                    record["next_check"] = max(record["first_seen"] + 120, time.time())
                if "follow_exit_code" not in record and not record.get("follow_process_closed"):
                    adopted.add(index)
            save()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            while True:
                ledger = json.loads((scratch / "20261104-subset-ledger.json").read_text())
                for job_name, cell in ledger["jobs"].items():
                    if f"-subset-{args.stage}-" not in job_name or cell["mode"] != args.mode:
                        continue
                    index = str(cell["index"])
                    if not cell.get("launched") or index in state:
                        continue
                    record = dict(
                        job_id=cell["job_id"],
                        first_seen=time.time(),
                        next_check=time.time() + 120,
                        terminal=False,
                        follow_started=True,
                        owner_pid=os.getpid(),
                    )
                    state[index] = record
                    save()  # Never create a second follower after an ambiguous local interruption.
                    log = scratch / f"{prefix}-{index}-follow.log"
                    with log.open("w") as handle:
                        follow = subprocess.Popen(
                            [*base, "job", "logs", cell["job_id"], "--follow"],
                            cwd=args.worktree,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    followers[index] = follow
                    record.update(follow_pid=follow.pid, follow_log=str(log))
                    save()
                for index, future in list(futures.items()):
                    if future.done():
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = {"error": repr(exc)}
                        record = state[index]
                        record["latest"] = result
                        record["next_check"] = time.time() + args.interval
                        if "job" in result:
                            record["terminal"] = result["job"]["state"] in {
                                "JOB_STATE_SUCCEEDED",
                                "JOB_STATE_FAILED",
                                "JOB_STATE_KILLED",
                            }
                        del futures[index]
                        save()
                for index, record in state.items():
                    if not record["terminal"] and index not in futures and time.time() >= record["next_check"]:
                        futures[index] = pool.submit(check, index, record["job_id"])
                    if index in followers and followers[index].poll() is not None:
                        record["follow_exit_code"] = followers.pop(index).returncode
                        save()
                for index in list(adopted):
                    record = state[index]
                    result = subprocess.run(
                        ["ps", "-p", str(record["follow_pid"]), "-o", "command="], capture_output=True, text=True
                    )
                    if result.returncode or record["job_id"] not in result.stdout:
                        record["follow_process_closed"] = True
                        record["follow_exit_observation"] = "Process gone after observer handoff; exit code unavailable"
                        adopted.remove(index)
                        save()
                if (
                    len(state) == expected
                    and all(r["terminal"] for r in state.values())
                    and not followers
                    and not adopted
                ):
                    return  # Scientific completion still requires W&B, metrics and checkpoint verification.
                time.sleep(5)


if __name__ == "__main__":
    main()
