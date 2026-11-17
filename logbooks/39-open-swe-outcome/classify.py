# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import collections

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3

configure_coreweave_s3()
V = "s3://marin-us-east-08a/marin/users/dml/open-swe-outcome/val-loss/v1"
fs = fsspec.filesystem("s3")
rows = pa.concat_tables([pq.read_table(fs.open(p)) for p in fs.glob(f"{V}/scores/*.parquet")]).to_pylist()
d = collections.defaultdict(dict)
for r in rows:
    if r["subset"].startswith("heldout"):
        d[r["trajectory_id"]][r["model"]] = r
tids = sorted(d)
y = np.array([d[t]["base"]["subset"] == "heldout-resolved" for t in tids])
part = np.array([d[t]["base"]["partition"] for t in tids])
ntok = np.array([d[t]["base"]["n_loss_tokens"] for t in tids], float)
nll = {
    m: np.array([d[t][m]["sum_nll"] for t in tids]) for m in ["base", "solved-s0", "solved-s1", "random-s0", "random-s1"]
}


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y.sum()
    return (ranks[y].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos))


rng = np.random.default_rng(0)


def report(name, score):
    a = auc(score, y)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        boots.append(auc(score[i], y[i]))
    within = []
    for p in sorted(set(part)):
        m = part == p
        if y[m].sum() >= 5 and (~y[m]).sum() >= 5:
            within.append((m.sum(), auc(score[m], y[m])))
    w = sum(n * a_ for n, a_ in within) / sum(n for n, _ in within)
    print(
        f"{name:<44} AUROC {a:.3f} [{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]"
        f"  within-partition {w:.3f}"
    )


print(f"n={len(y)} resolved={y.sum()}")
report("length: fewer assistant tokens", -ntok)
for m in nll:
    report(f"{m}: per-token loss (lower = resolved)", -nll[m] / ntok)
for s in ("s0", "s1"):
    report(f"ratio {s}: total log p_solved - log p_random", nll[f"random-{s}"] - nll[f"solved-{s}"])
    report(f"ratio {s}: per-token", (nll[f"random-{s}"] - nll[f"solved-{s}"]) / ntok)
report("ratio both seeds: total", nll["random-s0"] + nll["random-s1"] - nll["solved-s0"] - nll["solved-s1"])
report("ratio both seeds: per-token", (nll["random-s0"] + nll["random-s1"] - nll["solved-s0"] - nll["solved-s1"]) / ntok)
print(
    "seed agreement of per-token ratio (Pearson r):",
    round(
        np.corrcoef((nll["random-s0"] - nll["solved-s0"]) / ntok, (nll["random-s1"] - nll["solved-s1"]) / ntok)[0, 1], 3
    ),
)

ratio = (nll["random-s0"] + nll["random-s1"] - nll["solved-s0"] - nll["solved-s1"]) / ntok
print("corr(ratio, log length):", round(np.corrcoef(ratio, np.log(ntok))[0, 1], 3))
q = np.quantile(ntok, [0.25, 0.5, 0.75])
bins = np.digitize(ntok, q)
print(
    "ratio AUROC within length quartiles:",
    [round(auc(ratio[bins == b], y[bins == b]), 3) for b in range(4)],
    "resolved share per quartile:",
    [round(y[bins == b].mean(), 2) for b in range(4)],
)
strata = [(p, b) for p in set(part) for b in range(4)]
num = den = 0
for p, b in strata:
    m = (part == p) & (bins == b)
    if y[m].sum() >= 3 and (~y[m]).sum() >= 3:
        num += m.sum() * auc(ratio[m], y[m])
        den += m.sum()
print(f"ratio AUROC within partition x length-quartile strata: {num / den:.3f} (covering {den} of {len(y)})")
