# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import collections
import math

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3

configure_coreweave_s3()
V = "s3://marin-us-east-08a/marin/users/dml/open-swe-outcome/val-loss/v1"
MODELS = ["solved-s0", "solved-s1", "random-s0", "random-s1"]
fs = fsspec.filesystem("s3")


def load(subdir):
    rows = pa.concat_tables([pq.read_table(fs.open(f"{V}/{subdir}/{m}.parquet")) for m in MODELS]).to_pylist()
    out = collections.defaultdict(dict)
    for r in rows:
        if r["subset"].startswith("heldout"):
            out[r["trajectory_id"]][r["model"]] = r
    return out


turns = load("turn-scores")
totals = load("scores")
tids = sorted(turns)
rel = [abs(turns[t][m]["sum_nll"] - totals[t][m]["sum_nll"]) / totals[t][m]["sum_nll"] for t in tids for m in MODELS]
print(f"per-turn totals vs earlier scores: max relative difference {max(rel):.2e}")

y = np.array([turns[t]["solved-s0"]["subset"] == "heldout-resolved" for t in tids])
part = np.array([turns[t]["solved-s0"]["partition"] for t in tids])
n_turns = np.array([len(turns[t]["solved-s0"]["turn_tokens"]) for t in tids])
total_tokens = np.array([sum(turns[t]["solved-s0"]["turn_tokens"]) for t in tids], float)
length_bin = np.digitize(total_tokens, np.quantile(total_tokens, [0.25, 0.5, 0.75]))
print(f"n={len(y)} resolved={y.sum()} assistant turns: median {np.median(n_turns):.0f}, min {n_turns.min()}")


def auc(score, labels):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = labels.sum()
    return (ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * (len(labels) - pos))


rng = np.random.default_rng(0)


def report(name, select):
    """select(n) -> slice of turn indices to score."""
    ratio = np.empty(len(tids))
    for i, t in enumerate(tids):
        sl = select(n_turns[i])
        tok = sum(turns[t]["solved-s0"]["turn_tokens"][sl])
        nll = {m: sum(turns[t][m]["turn_nll"][sl]) for m in MODELS}
        ratio[i] = (nll["random-s0"] + nll["random-s1"] - nll["solved-s0"] - nll["solved-s1"]) / max(tok, 1)
    boots = [auc(ratio[j], y[j]) for j in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
    num = den = 0
    for p in set(part):
        for b in range(4):
            m = (part == p) & (length_bin == b)
            if y[m].sum() >= 3 and (~y[m]).sum() >= 3:
                num += m.sum() * auc(ratio[m], y[m])
                den += m.sum()
    print(
        f"{name:<28} AUROC {auc(ratio, y):.3f} [{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]"
        f"  within partition x length {num / den:.3f}"
    )


for f in (0.1, 0.25, 0.5, 0.75, 1.0):
    report(f"first {f:.0%} of turns", lambda n, f=f: slice(0, max(1, math.ceil(f * n))))
for k in (1, 3, 5):
    report(f"all but last {k} turns", lambda n, k=k: slice(0, max(1, n - k)))
for k in (1, 3):
    report(f"last {k} turns only", lambda n, k=k: slice(max(0, n - k), n))
