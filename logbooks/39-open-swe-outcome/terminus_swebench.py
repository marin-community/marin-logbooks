# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-trial grades and task-by-task comparisons for the Terminus-2 round of #39.

Harbor fails a run's final archiving step on a single incomplete trial, leaving the run record without
metrics, so this reads every trial's ``result.json`` directly.

    uv run python logbooks/39-open-swe-outcome/terminus_swebench.py
"""

import collections
import json
from concurrent.futures import ThreadPoolExecutor
from math import comb

import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3

EVALS = "marin-us-east-02a/marin/evals"
RUNS = {
    "base": "open-swe-terminus-eval-base-r3-swebench-verified-200-c9-declarative-0171ef9ccecd5984bd4ad2ed0aa1ff83",
    "solved-s0": (
        "open-swe-terminus-eval-solved-s0-r2-swebench-verified-200-c9-declarative-bece3e40a7355ae6896e7650163bafca"
    ),
    "random-s0": (
        "open-swe-terminus-eval-random-s0-r2-swebench-verified-200-c9-declarative-376dd7cb50725fecbbeaebea9672f58b"
    ),
}
AGENT_TIMEOUT = "AgentTimeoutError"


def trial_grades(fs, run: str) -> list[dict]:
    job = sorted(fs.ls(f"{EVALS}/{run}/results/harbor_jobs", detail=False))[-1]

    def read(path: str) -> dict:
        result = json.loads(fs.cat(path))
        verifier = result.get("verifier_result") or {}
        return {
            "task": result["task_name"],
            "reward": (verifier.get("rewards") or {}).get("reward"),
            "exception": (result.get("exception_info") or {}).get("exception_type"),
        }

    with ThreadPoolExecutor(32) as pool:
        return list(pool.map(read, fs.glob(f"{job}/*/attempts/000/result.json")))


def sign_test(a: int, b: int) -> float:
    """Two-sided exact sign test on discordant counts."""
    n = a + b
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2**n) if n else 1.0


def main() -> None:
    configure_coreweave_s3()
    fs = fsspec.filesystem("s3")
    trials = {name: trial_grades(fs, run) for name, run in RUNS.items()}
    solved = {
        name: {t["task"]: t["reward"] == 1 for t in rows if t["reward"] is not None} for name, rows in trials.items()
    }
    for name, rows in trials.items():
        exceptions = collections.Counter(t["exception"] for t in rows if t["exception"])
        timeouts = exceptions.pop(AGENT_TIMEOUT, 0)
        graded = solved[name]
        print(
            f"{name:<10} trials {len(rows)}  graded {len(graded)}  solved {sum(graded.values())}"
            f" ({sum(graded.values()) / len(graded):.1%})  agent timeouts {timeouts}  other exceptions {sum(exceptions.values())}"
        )
    common = set.intersection(*(set(s) for s in solved.values()))
    print(f"graded in all runs: {len(common)} tasks;", {n: sum(s[t] for t in common) for n, s in solved.items()})
    for a, b in (("solved-s0", "random-s0"), ("solved-s0", "base"), ("random-s0", "base")):
        shared = set(solved[a]) & set(solved[b])
        only_a = sum(solved[a][t] and not solved[b][t] for t in shared)
        only_b = sum(solved[b][t] and not solved[a][t] for t in shared)
        both = sum(solved[a][t] and solved[b][t] for t in shared)
        print(
            f"{a} vs {b}: {len(shared)} tasks; {a} only {only_a}, {b} only {only_b}, both {both};"
            f" sign test p={sign_test(only_a, only_b):.2g}"
        )


if __name__ == "__main__":
    main()
