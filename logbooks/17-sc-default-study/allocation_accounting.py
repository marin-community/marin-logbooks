# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combine retained allocation proofs without counting a task attempt twice."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def hours(start: str, end: str, gpus: int) -> float:
    elapsed = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    if elapsed < 0 or gpus <= 0:
        raise ValueError("Invalid allocation interval")
    return elapsed * gpus / 3600


def combine(paths: list[Path]) -> dict:
    jobs = []
    quota = []
    inputs = []
    attempts = set()
    job_ids = set()
    for path in paths:
        raw = path.read_bytes()
        proof = json.loads(raw)
        inputs.append({"path": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
        if "jobs" in proof:
            additions = proof["jobs"]
            quota.extend(proof["preexecution_reservation_episodes"])
        else:
            interrupted = "allocation" in proof
            rows = proof["allocation"] if interrupted else proof["tasks"]
            grouped = defaultdict(list)
            for row in rows:
                # This recorded preemption concerns five eight-H100 nodes.
                gpus = 8 if interrupted else row["gpus"]
                grouped[row["task_id"].rsplit("/", 1)[0]].append(
                    {
                        "task_id": row["task_id"],
                        "attempt_uid": row["attempt_uid"],
                        "gpus": gpus,
                        "start_utc": row["pod_scheduled"],
                        "end_utc": row.get("accounted_end", row.get("controller_terminal_utc")),
                        "H100_hours": row["H100_hours"],
                    }
                )
            additions = [{"job_id": key, "tasks": rows} for key, rows in grouped.items()]
            reported = proof["reconstructed_H100_hours_total"] if interrupted else proof["total_H100_hours"]
            if not math.isclose(sum(row["H100_hours"] for rows in grouped.values() for row in rows), reported):
                raise ValueError(f"Published proof total differs: {path}")
        for job in additions:
            if job["job_id"] in job_ids:
                raise ValueError(f"Repeated allocation job: {job['job_id']}")
            job_ids.add(job["job_id"])
            total = 0.0
            for row in job["tasks"]:
                identity = (row["task_id"], row["attempt_uid"])
                if identity in attempts:
                    raise ValueError(f"Repeated allocation attempt: {identity}")
                attempts.add(identity)
                calculated = hours(row["start_utc"], row["end_utc"], row["gpus"])
                if not math.isclose(calculated, row["H100_hours"], abs_tol=1e-8):
                    raise ValueError(f"Allocation clock differs from reported hours: {identity}")
                total += calculated
            jobs.append({**job, "source": path.name, "reconstructed_H100_hours": total})
    quota_ids = [(row["task_id"], row["attempt_uid"]) for row in quota]
    if len(set(quota_ids)) != len(quota_ids) or set(quota_ids) & attempts:
        raise ValueError("A preexecution quota episode was counted more than once")
    executed = sum(job["reconstructed_H100_hours"] for job in jobs)
    reserved = sum(row["observed_admission_to_preemption_quota_H100_hours"] for row in quota)
    return {
        "method": "Sum the retained per-attempt GPU allocation intervals; reject duplicate jobs or attempts.",
        "limits": (
            "Reconstructed spans, not exact billing or certified bounds. Starts use retained PodScheduled "
            "clocks or explicitly labeled startup/admission proxies. Ends retain controller-terminal, "
            "container-exit or retry/kill proxies as documented in each input. Unbound requeue gaps are "
            "excluded where separable; initial queue/startup windows included by a proxy are identified "
            "in the underlying proof. Preexecution quota reservation is separate from established physical GPU use. "
            "This report covers only the supplied terminal proofs; active jobs require later proofs."
        ),
        "inputs": inputs,
        "jobs": jobs,
        "job_count": len(jobs),
        "executed_attempt_count": len(attempts),
        "reconstructed_executed_task_H100_hours": executed,
        "observed_preexecution_quota_H100_hours": reserved,
        "reconstructed_tasks_plus_observed_quota_H100_hours": executed + reserved,
        "preexecution_reservation_episodes": quota,
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("proofs", type=Path, nargs="+")
    args = parser.parse_args()
    report = combine(args.proofs)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key.endswith(("_hours", "_count"))}))
