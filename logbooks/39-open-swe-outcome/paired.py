# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import collections
import gzip
import json
import math

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3

configure_coreweave_s3()
V = "s3://marin-us-east-08a/marin/users/dml/open-swe-outcome/val-loss/v2-paired"
MODELS = ["solved-s0", "solved-s1", "random-s0", "random-s1"]
fs = fsspec.filesystem("s3")

with fs.open(f"{V}/val.jsonl.gz", "rb") as raw, gzip.open(raw, "rt") as lines:
    pair_of = {}
    for line in lines:
        r = json.loads(line)
        pair_of[r["trajectory_id"]] = (r["partition"], r["instance_id"])

rows = pa.concat_tables([pq.read_table(fs.open(f"{V}/turn-scores/{m}.parquet")) for m in MODELS]).to_pylist()
by_tid = collections.defaultdict(dict)
for r in rows:
    by_tid[r["trajectory_id"]][r["model"]] = r
pairs = collections.defaultdict(dict)
for tid, models in by_tid.items():
    pairs[pair_of[tid]][models["solved-s0"]["subset"]] = models
pairs = {k: v for k, v in pairs.items() if len(v) == 2}
keys = sorted(pairs)
print(f"pairs: {len(keys)}; by partition: {dict(collections.Counter(k[0].split('open_swe_traces/')[1] for k in keys))}")


def ratio(models, select):
    n = len(models["solved-s0"]["turn_tokens"])
    sl = select(n)
    tok = sum(models["solved-s0"]["turn_tokens"][sl])
    nll = {m: sum(models[m]["turn_nll"][sl]) for m in MODELS}
    return (nll["random-s0"] + nll["random-s1"] - nll["solved-s0"] - nll["solved-s1"]) / max(tok, 1)


rng = np.random.default_rng(0)


def report(name, select):
    pos = np.array([ratio(pairs[k]["heldout-resolved"], select) for k in keys])
    neg = np.array([ratio(pairs[k]["heldout-unresolved"], select) for k in keys])
    wins = (pos > neg).astype(float) + 0.5 * (pos == neg)
    boots = [wins[rng.integers(0, len(wins), len(wins))].mean() for _ in range(2000)]
    print(
        f"{name:<24} within-pair win rate {wins.mean():.3f}"
        f" [{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]"
    )


for f in (0.1, 0.25, 0.5, 0.75, 1.0):
    report(f"first {f:.0%} of turns", lambda n, f=f: slice(0, max(1, math.ceil(f * n))))
for k in (1, 5):
    report(f"all but last {k} turns", lambda n, k=k: slice(0, max(1, n - k)))
report("last 1 turn only", lambda n: slice(n - 1, n))

full = lambda n: slice(0, n)  # noqa: E731
lengths = np.array([sum(pairs[k]["heldout-resolved"]["solved-s0"]["turn_tokens"]) for k in keys]), np.array(
    [sum(pairs[k]["heldout-unresolved"]["solved-s0"]["turn_tokens"]) for k in keys]
)
print(f"resolved trajectory shorter within pair: {(lengths[0] < lengths[1]).mean():.3f}")
