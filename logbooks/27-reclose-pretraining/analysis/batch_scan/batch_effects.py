# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Do per-step probe changes shared by both seeds track batch content? Usage: batch_effects.py SCAN_JSONL CURVES_CSV"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

scan = {r["step"]: r for r in map(json.loads, Path(sys.argv[1]).read_text().splitlines())}
curves = defaultdict(dict)
for r in csv.DictReader(Path(sys.argv[2]).read_text().splitlines()):
    curves[(r["run"], r["group"])][int(r["step"])] = (float(r["p_ge1"]), float(r["hazard"]))
runs = sorted({k[0] for k in curves})
groups = ("swe", "other+tool", "other-plain")


def feats(r):
    tt = r["trained_tokens"]
    return {
        "think_turns": r["turn_has_think"],
        "think_tok_frac": r["think_tokens"] / tt,
        "post_close_tok_frac": r["post_close_tokens"] / tt,
        "nothink_asst_tok_frac": r["nothink_asst_tokens"] / tt,
        "post_close_per_close": r["post_close_tokens"] / max(1, r["closes_trained"]),
        "think_per_close": r["think_tokens"] / max(1, r["closes_trained"]),
        "tool_turns": r.get("other_role_turns", 0),
        "nothink_turn_frac": 1 - r["turn_has_think"] / r["asst_turns"],
        "ge2_close_turns": r["turn_ge2_close"],
    }


steps = sorted(scan)
F = {s: feats(scan[s]) for s in steps}
names = list(F[steps[0]])
print(
    "feature ranges over steps:",
    {n: (round(min(F[s][n] for s in steps), 4), round(max(F[s][n] for s in steps), 4)) for n in names},
)
for lo in (0, 30):
    print(f"\n=== batch k vs change from probe k to k+1, k >= {lo} ===")
    for g in groups:
        for m, (idx, tf) in {"dP>=1": (0, lambda x: x), "dlog_hazard": (1, np.log)}.items():
            ks = [k for k in steps if k >= lo and all(k in curves[(r, g)] and k + 1 in curves[(r, g)] for r in runs)]
            d = {r: np.array([tf(curves[(r, g)][k + 1][idx]) - tf(curves[(r, g)][k][idx]) for k in ks]) for r in runs}
            agree = pearsonr(d[runs[0]], d[runs[1]])[0] if len(runs) == 2 else float("nan")
            shared = np.mean([d[r] for r in runs], axis=0)
            line = []
            for n in names:
                rho, p = spearmanr([F[k][n] for k in ks], shared)
                if abs(rho) >= 0.3 or n in ("post_close_tok_frac",):
                    line.append(f"{n} {rho:+.2f}{'*' if p < 0.01 else ''}")
            print(f"  {g:<12}{m:<12} n={len(ks):<3} seeds agree r={agree:+.2f} | " + "  ".join(line))
