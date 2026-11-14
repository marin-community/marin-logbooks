# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trough-to-peak normalized learning steps for closing vs re-closing, per run and group."""
import csv
import json
from pathlib import Path

A = Path(__file__).resolve().parents[1]
GROUPS = {"swe": "swe", "other+tool": "tool", "other-plain": "plain"}
LEVELS = (0.1, 0.5, 0.9)


def crossings(series):
    """Steps where (v - trough) / (peak - trough) first reaches each level, over the largest rise."""
    steps = sorted(series)
    vals = [series[s] for s in steps]
    # The largest rise: the trough/peak pair (trough first) with the biggest gain.
    best, trough_i, peak_i, low_i = -1.0, 0, 0, 0
    for j in range(len(vals)):
        if vals[j] < vals[low_i]:
            low_i = j
        if vals[j] - vals[low_i] > best:
            best, trough_i, peak_i = vals[j] - vals[low_i], low_i, j
    lo, hi = vals[trough_i], vals[peak_i]
    out = []
    for level in LEVELS:
        target = lo + level * (hi - lo)
        out.append(next(steps[i] for i in range(trough_i, peak_i + 1) if vals[i] >= target))
    return out, steps[trough_i], steps[peak_i], lo, hi


def table(name, close, reclose):
    rows = []
    for g in ("swe", "tool", "plain"):
        if g not in close:
            continue
        c, ct, cp, clo, chi = crossings(close[g])
        r, rt, rp, rlo, rhi = crossings(reclose[g])
        rows.append((name, g, c, r, (ct, cp, clo, chi), (rt, rp, rlo, rhi)))
    return rows


series = {}
# Old probe: p_first_close (close at 10.05's close point) and p_reclose (close again at 10.05's re-close point).
for run, f in (
    ("answer (old probe)", "probe_replay-10400-base300e.csv"),
    ("mean (old probe)", "probe_replay-10400-meanseed.csv"),
):
    pf, pr = {}, {}
    for r in csv.DictReader(open(A / f)):
        g = GROUPS[r["group"]]
        s = int(r["step"])
        pf.setdefault(g, {})[s] = float(r["p_first_close"])
        pr.setdefault(g, {})[s] = float(r["p_reclose"])
    series[run] = (pf, pr, "p_first_close", "p_reclose")
D = json.load(open(Path(__file__).parent / "page_data.json"))
for key, name in (("answer", "answer (est)"), ("mean", "mean (est)"), ("ltslash", "</ (est, 20 steps)")):
    est = D["est"][key]
    series[name] = (
        {g: {p[0]: p[1] for p in v} for g, v in est.items()},
        {g: {p[0]: p[2] for p in v} for g, v in est.items()},
        "P(>=1)",
        "P(>=2)",
    )
for key, name in (("answer", "answer (exact)"), ("ltslash", "</ (exact)")):
    ex = D["exact"][key]
    series[name] = (
        {g: {p[0]: p[1] for p in v} for g, v in ex.items()},
        {g: {p[0]: p[2] for p in v} for g, v in ex.items()},
        "P(>=1)",
        "hazard",
    )

print(
    f"{'run':22} {'grp':5} {'close metric':13} {'10/50/90%':14} {'re-close metric':15} {'10/50/90%':14} {'lag at 50%':>10}"
)
results = []
for name, (cl, rc, cm, rm) in series.items():
    for row in table(name, cl, rc):
        _, g, c, r, ci, ri = row
        print(
            f"{name:22} {g:5} {cm:13} {c!s:14} {rm:15} {r!s:14} {r[1] - c[1]:>+10}   close trough->peak s{ci[0]}->s{ci[1]} ({ci[2]:.3g}->{ci[3]:.3g}); re-close s{ri[0]}->s{ri[1]} ({ri[2]:.3g}->{ri[3]:.3g})"
        )
        results.append(
            {
                "run": name,
                "group": g,
                "close_metric": cm,
                "close": c,
                "reclose_metric": rm,
                "reclose": r,
                "close_range": ci,
                "reclose_range": ri,
            }
        )
json.dump(results, open(Path(__file__).parent / "trough_peak.json", "w"))
