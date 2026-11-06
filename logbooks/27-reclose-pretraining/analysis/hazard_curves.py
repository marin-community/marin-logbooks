# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""P(close on time), P(close early) and P(at least two closes) given ``<|start_think|>``, per replay step.

Usage: hazard_curves.py TRAJECTORIES_JSONL OUT_PREFIX NAME=DIR [NAME=DIR ...]

Each DIR holds a replay's per-trajectory probe dumps (``step-N.jsonl`` from ``reclose_eval.py``). On the fixed
10.05 trajectories, teacher-forced:

* ``early = 1 - (1 - p_premature)^L``: the chance of closing before 10.05 did, where ``L`` is the reasoning
  length and ``p_premature`` the mean close probability over it. The dumps keep only that mean, so this is a
  lower bound on ``1 - prod(1 - p)`` (AM-GM) and the survival term below is an upper bound.
* ``on_time = (1 - p_premature)^L * p_first_close``: the chance the first close lands where 10.05's did.
* ``double = on_time * hazard``: the chance of closing on time and then again in 10.05's continuation.

Writes ``OUT_PREFIX.csv``, ``OUT_PREFIX.png``, and prints the first step where ``on_time`` reaches
``ON_TIME_LEARNED`` (k) and no-lead-in ``double`` reaches ``DOUBLE_LEARNED`` (j) for each run and group.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIRST_STEP = 157000
ON_TIME_LEARNED = 0.5
DOUBLE_LEARNED = 0.05
METRICS = ("on_time", "early", "double")
GROUPS = {"swe": "SWE", "other+tool": "Tool-using", "other-plain": "Plain"}
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e6e5e0"


def key(row: dict) -> tuple:
    return row["group"], row["id"], row["sample"], row["bridge"]


def step_cells(path: Path, lengths: dict[tuple, int]) -> dict[str, dict[str, float]]:
    cells = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            survive = (1.0 - row["p_premature"]) ** lengths[key(row)]
            on_time = survive * row["p_first_close"]
            lead = "lead_in" if row["bridge"] else "no_lead_in"
            cells[f"{row['group']}/{lead}"].append((on_time, 1.0 - survive, on_time * row["hazard"]))
    return {cell: dict(zip(METRICS, np.mean(values, axis=0), strict=True)) for cell, values in cells.items()}


def first_step(series: dict[int, float], threshold: float) -> int | None:
    return next((s for s in sorted(series) if series[s] >= threshold), None)


def main() -> None:
    with open(sys.argv[1]) as handle:
        lengths = {key(t): t["close"] - t["reasoning_start"] for t in map(json.loads, handle)}
    out = sys.argv[2]
    runs = {}
    for spec in sys.argv[3:]:
        name, directory = spec.split("=", 1)
        runs[name] = {
            int(p.stem.split("-")[1]) - FIRST_STEP: step_cells(p, lengths)
            for p in sorted(Path(directory).glob("step-*.jsonl"))
        }
    with open(f"{out}.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "step", "cell", *METRICS])
        for name, steps in runs.items():
            for step, cells in sorted(steps.items()):
                for cell, m in sorted(cells.items()):
                    writer.writerow([name, step, cell, *(m[k] for k in METRICS)])

    print(
        f"k: first step with on-time close >= {ON_TIME_LEARNED}; j: first step with no-lead-in double close >="
        f" {DOUBLE_LEARNED} (lead-in j in brackets); peak double close (step)"
    )
    for name, steps in runs.items():
        parts = []
        for group in GROUPS:
            on_time = {s: c[f"{group}/no_lead_in"]["on_time"] for s, c in steps.items()}
            double = {s: c[f"{group}/no_lead_in"]["double"] for s, c in steps.items()}
            double_lead = {s: c[f"{group}/lead_in"]["double"] for s, c in steps.items()}
            peak = max(double, key=double.get)
            parts.append(
                f"{group} k={first_step(on_time, ON_TIME_LEARNED)} j={first_step(double, DOUBLE_LEARNED)}"
                f" [{first_step(double_lead, DOUBLE_LEARNED)}] peak={double[peak]:.3f} ({peak})"
            )
        print(f"  {name:<14}" + "   ".join(parts) + f"   (last step {max(steps)})")

    rows = (
        ("no_lead_in", "on_time", "P(first close on time | start)", "linear", ON_TIME_LEARNED),
        ("no_lead_in", "early", "P(close early | start)", "linear", None),
        ("no_lead_in", "double", "P(≥2 closes | start)", "log", DOUBLE_LEARNED),
        ("lead_in", "double", 'P(≥2 closes | start),\nforced "Wait,"-style lead-in', "log", DOUBLE_LEARNED),
    )
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK})
    fig, axes = plt.subplots(len(rows), len(GROUPS), figsize=(13, 12.5), sharex=True, facecolor=SURFACE)
    for col, (group, title) in enumerate(GROUPS.items()):
        for r, (lead, metric, label, scale, threshold) in enumerate(rows):
            ax = axes[r][col]
            ax.set_facecolor(SURFACE)
            for (name, steps), color in zip(runs.items(), COLORS, strict=False):
                xs = sorted(steps)
                ys = [steps[s][f"{group}/{lead}"][metric] for s in xs]
                ax.plot(xs, ys, color=color, linewidth=2, label=name)
                ax.plot(xs[-1], ys[-1], "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
            if threshold is not None:
                ax.axhline(threshold, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
            ax.set_yscale(scale)
            if scale == "linear":
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(1e-4, 1)
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.tick_params(colors=MUTED)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=12, loc="left")
            if col == 0:
                ax.set_ylabel(label)
            if r == len(rows) - 1:
                ax.set_xlabel("SFT replay step")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(runs), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("When the replay learns to close, and to close twice", y=1.035, fontsize=13)
    fig.text(
        0.5,
        -0.01,
        "Teacher-forced on fixed 10.05 trajectories. On time: the first close lands where 10.05 closed."
        f" Dashed lines: {ON_TIME_LEARNED} (k) and {DOUBLE_LEARNED} (j).",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=130, bbox_inches="tight", facecolor=SURFACE)


if __name__ == "__main__":
    main()
