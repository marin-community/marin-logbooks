"""Measure lower bounds on individual requests from the public Qwen trace counters."""

import json
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from math import ceil
from urllib.request import urlopen

REVISION = "f16153e5c47c46b2f6f2ebf0aaefae34bdf8c8bd"
ROOT = f"https://raw.githubusercontent.com/swescience/swescience.github.io/{REVISION}"
TRACE = ROOT + "/public/traces/qwen3-8-27b-xhigh/task-{:03d}.json"


def usage(task: int) -> tuple[int, dict]:
    with urlopen(TRACE.format(task), timeout=30) as response:
        trace = json.load(response)
    assert trace["run"]["id"] == f"qwen3-8-27b-xhigh-{task:03d}"
    return task, trace["usage"]


def main() -> None:
    with ThreadPoolExecutor(max_workers=2) as workers:
        rows = dict(workers.map(usage, range(1, 120)))
    for field in ("cacheReadTokens", "outputTokens"):
        # Using cached input alone avoids assumptions about whether inputTokens includes cache reads.
        averages = {
            task: Fraction(row[field], row["callCount"])
            for task, row in rows.items()
            if row[field] is not None and row["callCount"]
        }
        task = max(averages, key=averages.get)
        print(
            f"{field}: {len(averages)} reports; task {task:03d}: {rows[task][field]} tokens / "
            f"{rows[task]['callCount']} calls; at least one call >= {ceil(averages[task])} tokens."
        )


if __name__ == "__main__":
    main()
