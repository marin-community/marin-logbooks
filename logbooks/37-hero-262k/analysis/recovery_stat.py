"""Paired sign-flip randomization test of MRCR recovery and full NLL between arms, with Holm correction."""
import itertools
import json
import sys
from collections import defaultdict

import numpy as np

ROOT = sys.argv[1]  # the results/mrcr directory
ARMS = ["keep166", "keep166-raw", "global188", "global200", "log32768", "log65536a20", "log65536sq"]
CANDS = ["global188", "global200", "log32768"]
FLIPS = 20000
rng = np.random.default_rng(20261106)


def load(arm):
    rows = {}
    for line in open(f"{ROOT}/{arm}/mrcr_example_losses.jsonl"):
        r = json.loads(line)
        rows[r["source_id"]] = r
    return rows


data = {a: load(a) for a in ARMS}
ids = sorted(data["keep166"])
assert all(sorted(d) == ids for d in data.values())
first = data["keep166"]
meta = [(first[i]["n_needles"], first[i]["distance_band"]) for i in ids]
print("sources", len(ids), "fields", sorted(first[ids[0]]), file=sys.stderr)
for i in ids:
    assert all(data[a][i]["scored_tokens"] == first[i]["scored_tokens"] for a in ARMS)
tok = np.array([first[i]["scored_tokens"] for i in ids], float)
A = {
    a: np.array([[data[a][i][k] for k in ("full_context_loss_sum", "query_only_loss_sum", "needle_only_loss_sum", "distractor_only_loss_sum")] for i in ids])
    for a in ARMS
}


def rec(x):  # x[..., 4] sums
    return (x[..., 3] - x[..., 0]) / (x[..., 1] - x[..., 2])


def test(a, b, idx):
    xa, xb, t = A[a][idx], A[b][idx], tok[idx].sum()
    d_rec = rec(xa.sum(0)) - rec(xb.sum(0))
    d_full = (xa[:, 0].sum() - xb[:, 0].sum()) / t
    m = rng.integers(0, 2, size=(FLIPS, len(idx), 1)).astype(float)
    sa = (m * xa[None] + (1 - m) * xb[None]).sum(1)
    sb = (m * xb[None] + (1 - m) * xa[None]).sum(1)
    nr = rec(sa) - rec(sb)
    nf = (sa[:, 0] - sb[:, 0]) / t
    # paired bootstrap interval, resampling sources
    bi = rng.integers(0, len(idx), size=(10000, len(idx)))
    br = rec(xa[bi].sum(1)) - rec(xb[bi].sum(1))
    pr = (1 + (np.abs(nr) >= abs(d_rec)).sum()) / (FLIPS + 1)
    pf = (1 + (np.abs(nf) >= abs(d_full)).sum()) / (FLIPS + 1)
    return dict(a=a, b=b, n=len(idx), d_rec=float(d_rec), p_rec=float(pr), ci=[float(x) for x in np.percentile(br, [2.5, 97.5])], d_full=float(d_full), p_full=float(pf), ra=float(rec(xa.sum(0))), rb=float(rec(xb.sum(0))))


def holm(ps):
    order = np.argsort(ps)
    out, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


groups = defaultdict(list)
for k, (n, band) in enumerate(meta):
    groups["all"].append(k)
    groups[f"{n}needle"].append(k)
    groups[str(band)].append(k)
    groups[f"{n}needle/{band}"].append(k)
print({g: len(v) for g, v in groups.items()})

idx = np.array(groups["all"])
print("\nA. Overall, all 15 pairs of merged arms (Holm over 15), plus merged vs raw")
merged = [a for a in ARMS if a != "keep166-raw"]
res = [test(a, b, idx) for a, b in itertools.combinations(merged, 2)]
hr, hf = holm([r["p_rec"] for r in res]), holm([r["p_full"] for r in res])
for r, x, y in zip(res, hr, hf):
    print(f"{r['a']:12s}-{r['b']:12s} rec {r['d_rec']:+.4f} CI [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] p={r['p_rec']:.4f} holm={x:.3f} | full {r['d_full']:+.4f} p={r['p_full']:.4f} holm={y:.3f}")
r = test("keep166", "keep166-raw", idx)
print(f"merged-raw keep rec {r['d_rec']:+.4f} CI {r['ci']} p={r['p_rec']:.4f} | full {r['d_full']:+.4f} p={r['p_full']:.4f}")

print("\nB. Candidates vs Keep merged across groups with n>=25 (Holm over the family, each metric)")
gs = [g for g, v in groups.items() if len(v) >= 25]
res = [dict(test(c, "keep166", np.array(groups[g])), g=g) for g in gs for c in CANDS]
hr, hf = holm([r["p_rec"] for r in res]), holm([r["p_full"] for r in res])
print("family size", len(res), "raw p<.05 rec", sum(r["p_rec"] < 0.05 for r in res), "full", sum(r["p_full"] < 0.05 for r in res), "holm<.05 rec", sum(x < 0.05 for x in hr), "full", sum(x < 0.05 for x in hf))
for r, x, y in sorted(zip(res, hr, hf), key=lambda t: t[0]["p_rec"])[:10]:
    print(f"{r['g']:24s} n={r['n']:3d} {r['a']:10s} rec {r['d_rec']:+.4f} p={r['p_rec']:.4f} holm={x:.3f} | full {r['d_full']:+.4f} p={r['p_full']:.4f} holm={y:.3f}")
