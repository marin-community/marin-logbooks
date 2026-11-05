# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare exact terminal collection requests from saved owner snapshots; no RPCs."""
import hashlib
import json
from pathlib import Path

SOURCE = "2bca5a11292d8631a0d200c5a1f9bfcc66bb2ce8"
TERMINAL = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_KILLED",
    "JOB_STATE_WORKER_FAILED",
    "JOB_STATE_UNSCHEDULABLE",
}


def prepare(observers, ledger, completed, joins):
    validated = {r["index"]: r for r in completed["results"] if r["scientifically_complete"]}
    requests, waiting = [], []
    for key, entry in sorted(ledger["cells"].items(), key=lambda x: int(x[0])):
        index = int(key)
        observed = observers[key]
        job = observed["latest"]["job"]
        if entry["source_commit"] != SOURCE or not entry["launch_started"]:
            raise ValueError(f"Invalid launch binding for cell {index}")
        if entry["job_id"] != observed["job_id"] or job["job_id"] != entry["job_id"]:
            raise ValueError(f"Job identity mismatch for cell {index}")
        if index in validated:
            result = validated[index]
            if result["job_id"] != entry["job_id"] or result["source"] != SOURCE:
                raise ValueError(f"Completed identity mismatch for cell {index}")
            continue
        if job["state"] not in TERMINAL:
            waiting.append(index)
            continue
        if not observed["terminal"]:
            raise ValueError(f"Owner has not reconciled terminal cell {index}")
        count = job["task_count"]
        expected_count = 1 if entry["cell"]["alpha"] == 0 else 4
        if count != expected_count:
            raise ValueError(f"Unexpected task count for cell {index}")
        cached = joins.get("cells", {}).get(key)
        requests.append(
            {
                "index": index,
                "job_id": entry["job_id"],
                "source_commit": SOURCE,
                "canonical_evidence": observed["latest"]["path"],
                "canonical_state": job["state"],
                "failure_count": job["failure_count"],
                "preemption_count": job["preemption_count"],
                "task_ids": [f"{entry['job_id']}/{i}" for i in range(count)],
                "cached_startup_join": cached,
                "task_collection_policy": (
                    "Read each terminal task once into a new final-evidence directory. Enumerate every returned"
                    " attempt; retain failures and preemptions. Reuse existing final artifacts after identity "
                    "validation."
                ),
                "log_collection_policy": (
                    "Use cached matching attempt joins; collect missing attempt joins with bounded reads only. "
                    "Preserve log transport errors separately. Never start followers or repoll the terminal "
                    "job."
                ),
                "scientifically_complete": False,
                "required_gates": [
                    "all task attempts and canonical outcome",
                    "exact frozen configuration and data/source identity",
                    "7630 finite native metric rows and final C4",
                    "native stored counter7630 and permanent checkpoint metadata",
                    "ONLINE finish and exact task-attempt/run joins",
                    "full-state restoration evidence for any claimed resumed intervals",
                ],
            }
        )
    return {
        "source_commit": SOURCE,
        "validated_reused": sorted(validated),
        "waiting_cells": waiting,
        "terminal_collection_requests": requests,
        "scope": "Offline request preparation only. No RPC, W&B, storage read or scientific completion claim.",
    }


def main():
    root = Path.cwd()
    execution = root.parent / "ema-gram-subset-seed0-r02-20261105"
    paths = {
        "observers": execution / "scratch/20261105-subset-training-r02-observers.json",
        "ledger": execution / "scratch/20261105-subset-training-r02-ledger.json",
        "completed": root / "logbooks/20-ema-gram-polar/subset-training-r02-alpha0-completed.json",
        "joins": root / "scratch/20261105-r02-attempt-joins/validated.json",
    }
    raw = {key: path.read_bytes() for key, path in paths.items()}
    result = prepare(**{key: json.loads(value) for key, value in raw.items()})
    result["inputs"] = {
        key: {"path": str(paths[key]), "sha256": hashlib.sha256(value).hexdigest()} for key, value in raw.items()
    }
    output = root / "scratch/20261105-r02-terminal-collection-plan.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "reused": result["validated_reused"],
                "waiting": len(result["waiting_cells"]),
                "eligible": len(result["terminal_collection_requests"]),
            }
        )
    )


if __name__ == "__main__":
    main()
