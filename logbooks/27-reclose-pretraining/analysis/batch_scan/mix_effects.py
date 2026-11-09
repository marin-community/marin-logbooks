# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Spearman of per-step source shares against the probe change both seeds share.

Usage: mix_effects.py SOURCE_MIX_JSONL BATCH_SCAN_JSONL CURVES_CSV
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

mix = {r.pop("step"): r for r in map(json.loads, Path(sys.argv[1]).read_text().splitlines())}
scan = {r["step"]: r for r in map(json.loads, Path(sys.argv[2]).read_text().splitlines())}
srcs = ["chat_think", "chat_think_tools", "chat_nothink", "chat_nothink_tools", "raw_text", "continuation"]
F = {}
for k, r in mix.items():
    tot = sum(r.values())
    F[k] = {s: r.get(s, 0) / tot for s in srcs}
    F[k]["post_close_frac"] = scan[k]["post_close_tokens"] / tot
C = defaultdict(dict)
for r in csv.DictReader(Path(sys.argv[3]).read_text().splitlines()):
    C[(r["run"], r["group"])][int(r["step"])] = (float(r["p_ge1"]), float(r["hazard"]))
names = list(F[0])
for lo in (10, 30):
    print(f"\nshared change (probe k -> k+1) vs batch k share, Spearman, k >= {lo}  [lag1 = batch k-1]")
    print(f"  {'':<26}" + "".join(f"{n[:14]:>15}" for n in names))
    for g in ("swe", "other+tool", "other-plain"):
        for m, idx, tf in (("dP>=1", 0, lambda x: x), ("dlogH", 1, np.log)):
            ks = [k for k in range(1, 100) if k >= lo and all(k + 1 in C[(r, g)] for r in ("answer", "ltslash"))]
            d = np.array(
                [np.mean([tf(C[(r, g)][k + 1][idx]) - tf(C[(r, g)][k][idx]) for r in ("answer", "ltslash")]) for k in ks]
            )
            for lag in (0, 1):
                row = []
                for n in names:
                    rho, p = spearmanr([F[k - lag][n] for k in ks], d)
                    row.append(f"{rho:+.2f}{'*' if p < 0.01 else ' '}")
                print(f"  {g:<12}{m:<7}lag{lag}  n={len(ks):<3}" + "".join(f"{x:>15}" for x in row))
