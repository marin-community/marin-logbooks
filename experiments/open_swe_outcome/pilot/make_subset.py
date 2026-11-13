"""Write the 200-task SWE-bench Verified pilot policy and the matching snapshot-name list."""

import json
import random
import sys
from pathlib import Path

SEED = 20261113
N = 200
CONCURRENCY = 12

names = json.loads(Path(sys.argv[1]).read_text())["names"]
tasks = sorted(random.Random(SEED).sample(sorted(names), N))
policy = [
    "# 200 SWE-bench Verified tasks: random.Random(20261113).sample(sorted(all 500 task ids), 200).",
    "# Concurrency is 12 because the Daytona evaluation account allows 45 snapshots and each task",
    "# image is one snapshot; a cleaner deletes each task's snapshot after its trial finishes.",
    f"n_concurrent_trials: {CONCURRENCY}",
    "retry:",
    "  max_retries: 10",
    "  include_exceptions:",
    "    - InternalServerError",
    "    - APIConnectionError",
    "    - APITimeoutError",
    "  exclude_exceptions: []",
    "  wait_multiplier: 2.0",
    "  min_wait_sec: 1.0",
    "  max_wait_sec: 60.0",
    "environment:",
    "  type: daytona",
    "agents:",
    "  - name: terminus-2",
    "datasets:",
    "  - name: hf://DCAgent/swebench-verified",
    "    ref: main",
    "    task_names:",
    *[f"      - {task}" for task in tasks],
]
Path(sys.argv[2]).write_text("\n".join(policy) + "\n")
Path(sys.argv[3]).write_text(json.dumps({task: names[task] for task in tasks}, indent=1))
repos = {}
for task in tasks:
    repos[task.split("__")[0]] = repos.get(task.split("__")[0], 0) + 1
print(len(tasks), "tasks;", dict(sorted(repos.items(), key=lambda kv: -kv[1])))
