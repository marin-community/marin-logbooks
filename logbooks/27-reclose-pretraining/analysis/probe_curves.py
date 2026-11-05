"""Aggregate the per-step re-close probe files of an SFT replay run into one CSV (mean per group and metric).

Usage: uv run python probe_curves.py <run-name> <out.csv>
Reads s3://marin-us-east-02a/tmp/ttl=30d/dml/sft_replay_10400/reclose/<run>/step-*.jsonl (CoreWeave credentials required).
"""
import csv, json, re, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

ROOT = "s3://marin-us-east-02a/tmp/ttl=30d/dml/sft_replay_10400/reclose"
START_STEP = 157000
METRICS = ("p_first_close", "p_reclose", "hazard", "mean_p", "p_premature")

def aggregate(fs, path):
    with fs.open(path) as handle:
        rows = [json.loads(line) for line in handle]
    values = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric in METRICS:
            if isinstance(row.get(metric), (int, float)):
                values[row["group"]][metric].append(row[metric])
    return {group: {m: sum(v) / len(v) for m, v in d.items()} for group, d in values.items()}

def main(run, out):
    configure_coreweave_s3()
    fs, root = filesystem_for(f"{ROOT}/{run}")
    files = sorted(fs.ls(root, detail=False))
    steps = [int(re.search(r"step-(\d+)", f).group(1)) - START_STEP for f in files]
    with ThreadPoolExecutor(16) as pool:
        series = list(pool.map(lambda f: aggregate(fs, "s3://" + f), files))
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "group", *METRICS])
        for step, agg in zip(steps, series):
            for group, metrics in sorted(agg.items()):
                writer.writerow([step, group, *(f"{metrics.get(m, float('nan')):.6e}" for m in METRICS)])
    print(out, len(steps), "steps")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
