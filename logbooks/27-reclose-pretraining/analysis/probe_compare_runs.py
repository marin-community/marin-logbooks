# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare re-close probe curves of several replay runs from their per-trajectory ``step-N.jsonl`` dumps.

Usage: probe_compare_runs.py OUT_CSV NAME=DIR [NAME=DIR ...]

Writes one row per run, step, and cell (group, and group split by lead-in) with mean ``p_first_close``,
``p_reclose`` and ``hazard``; prints the no-lead-in hazard peak per group and a few steps side by side.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FIRST_STEP = 157000
SHOWN_STEPS = (0, 5, 8, 10, 12, 15, 20, 25, 30)


def cell_means(path: Path) -> dict[str, dict[str, float]]:
    cells = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            cells[row["group"]].append(row)
            cells[f"{row['group']}/{'lead_in' if row['bridge'] else 'no_lead_in'}"].append(row)
    out = {}
    for cell, rows in cells.items():
        recloses = [r["p_reclose"] for r in rows if r["p_reclose"] is not None]
        out[cell] = {
            "p_first_close": float(np.mean([r["p_first_close"] for r in rows])),
            "p_reclose": float(np.mean(recloses)) if recloses else float("nan"),
            "hazard": float(np.mean([r["hazard"] for r in rows])),
        }
    return out


def main() -> None:
    runs = {}
    for spec in sys.argv[2:]:
        name, directory = spec.split("=", 1)
        runs[name] = {
            int(p.stem.split("-")[1]) - FIRST_STEP: cell_means(p) for p in sorted(Path(directory).glob("step-*.jsonl"))
        }
    with open(sys.argv[1], "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "step", "cell", "p_first_close", "p_reclose", "hazard"])
        for name, steps in runs.items():
            for step, cells in sorted(steps.items()):
                for cell, m in sorted(cells.items()):
                    writer.writerow([name, step, cell, m["p_first_close"], m["p_reclose"], m["hazard"]])

    groups = ("swe", "other+tool", "other-plain")
    print("no-lead-in hazard peak (step) within each run's recorded steps")
    for name, steps in runs.items():
        peaks = []
        for g in groups:
            series = {s: c[f"{g}/no_lead_in"]["hazard"] for s, c in steps.items() if f"{g}/no_lead_in" in c}
            best = max(series, key=series.get)
            peaks.append(f"{g} {series[best]:.3f} ({best})")
        print(f"  {name:<12}" + "   ".join(peaks))
    for g in groups:
        print(f"\n{g}: p_first_close / p_reclose / no-lead-in hazard")
        print("  step " + "".join(f"{name:>28}" for name in runs))
        for step in SHOWN_STEPS:
            line = f"  {step:>4} "
            for steps in runs.values():
                c = steps.get(step)
                if c is None:
                    line += f"{'-':>28}"
                    continue
                line += f"{c[g]['p_first_close']:>9.4f}{c[g]['p_reclose']:>9.4f}{c[g + '/no_lead_in']['hazard']:>10.4f}"
            print(line)


if __name__ == "__main__":
    main()
