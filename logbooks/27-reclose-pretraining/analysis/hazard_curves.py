# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""When a replay learns to close thinking, and to close it again, per step.

Usage: hazard_curves.py TRAJECTORIES_JSONL OUT_PREFIX NAME=DIR [NAME=DIR ...]

Each DIR holds a replay's per-trajectory probe dumps (``step-N.jsonl`` from ``reclose_eval.py``). Both metrics
are teacher-forced along the fixed 10.05 trajectories without a lead-in, averaged per group:

* ``p_ge1``, P(thinking closed at least once | opened): every position from the first reasoning token through
  the end of the scored text is an independent chance to emit ``<|end_think|>``, so nothing assumes 10.05
  closed in the right place.
* ``hazard``, P(closed again | a close is already in the text): the chance of another close in the
  continuation after 10.05's close token. Closes earlier in the reasoning do not count here, because under
  teacher forcing the text never contains them.

Dumps written after commit 9893fb5798 carry the exact ``p_ge1``. Older dumps keep only the mean close
probability over the reasoning (``p_premature``, ``L`` positions), the probability at 10.05's close and the
continuation hazard; for those, ``p_ge1`` treats the reasoning as ``Binomial(L, mean)``, a lower bound (AM-GM).
``hazard`` is exact in every dump.

Writes ``OUT_PREFIX.csv`` and ``OUT_PREFIX.png`` and prints the first step where ``p_ge1`` reaches
``CLOSE_LEARNED`` (k) and ``hazard`` reaches ``RECLOSE_LEARNED`` (j) for each run and group.
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
CLOSE_LEARNED = 0.5
RECLOSE_LEARNED = 0.05
METRICS = ("p_ge1", "hazard")
GROUPS = {"swe": "SWE", "other+tool": "Tool-using", "other-plain": "Plain"}
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e6e5e0"


def key(row: dict) -> tuple:
    return row["group"], row["id"], row["sample"], row["bridge"]


def estimated_p_ge1(row: dict, reasoning_length: int) -> float:
    return 1.0 - (1.0 - row["p_premature"]) ** reasoning_length * (1.0 - row["p_first_close"]) * (1.0 - row["hazard"])


def step_means(path: Path, lengths: dict[tuple, int]) -> dict[str, dict[str, float]]:
    cells = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if row["bridge"]:
                continue
            p_ge1 = row["p_ge1"] if "p_ge1" in row else estimated_p_ge1(row, lengths[key(row)])
            cells[row["group"]].append((p_ge1, row["hazard"]))
    return {group: dict(zip(METRICS, np.mean(values, axis=0), strict=True)) for group, values in cells.items()}


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
            int(p.stem.split("-")[1]) - FIRST_STEP: step_means(p, lengths)
            for p in sorted(Path(directory).glob("step-*.jsonl"))
        }

    with open(f"{out}.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "step", "group", *METRICS])
        for name, steps in runs.items():
            for step, groups in sorted(steps.items()):
                for group, m in sorted(groups.items()):
                    writer.writerow([name, step, group, *(m[k] for k in METRICS)])

    print(f"k: first step with P(>=1 | opened) >= {CLOSE_LEARNED}; j: first step with hazard >= {RECLOSE_LEARNED}")
    for name, steps in runs.items():
        parts = []
        for group in GROUPS:
            series = {k: {s: g[group][k] for s, g in steps.items()} for k in METRICS}
            parts.append(
                f"{group} k={first_step(series['p_ge1'], CLOSE_LEARNED)} j={first_step(series['hazard'], RECLOSE_LEARNED)}"
            )
        print(f"  {name:<24}" + "   ".join(parts) + f"   (last step {max(steps)})")

    rows = (
        ("p_ge1", "P(closed ≥1 | opened)", "linear", CLOSE_LEARNED),
        ("hazard", "P(closes again |\na close is in the text)", "log", RECLOSE_LEARNED),
    )
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK})
    fig, axes = plt.subplots(len(rows), len(GROUPS), figsize=(13, 7), sharex=True, facecolor=SURFACE)
    for col, (group, title) in enumerate(GROUPS.items()):
        for r, (metric, label, scale, threshold) in enumerate(rows):
            ax = axes[r][col]
            ax.set_facecolor(SURFACE)
            for (name, steps), color in zip(runs.items(), COLORS, strict=False):
                xs = sorted(steps)
                ys = [steps[s][group][metric] for s in xs]
                ax.plot(xs, ys, color=color, linewidth=2, label=name)
                ax.plot(xs[-1], ys[-1], "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
            ax.axhline(threshold, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
            ax.set_yscale(scale)
            ax.set_ylim(*((0, 1) if scale == "linear" else (1e-4, 1)))
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
    fig.suptitle("When the replay learns to close thinking, and to close it again", y=1.035, fontsize=13)
    fig.text(
        0.5,
        -0.01,
        "Teacher-forced on fixed 10.05 trajectories without a lead-in. Top: a close anywhere. Bottom: a further"
        f" close after 10.05's close token. Dashed lines: {CLOSE_LEARNED} and {RECLOSE_LEARNED}.",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=130, bbox_inches="tight", facecolor=SURFACE)


if __name__ == "__main__":
    main()
