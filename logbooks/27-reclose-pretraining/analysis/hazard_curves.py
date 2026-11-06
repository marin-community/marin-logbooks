# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""P(thinking closed at least once | opened) and P(at least twice | opened), per replay step.

Usage: hazard_curves.py TRAJECTORIES_JSONL OUT_PREFIX NAME=DIR [NAME=DIR ...]

Each DIR holds a replay's per-trajectory probe dumps (``step-N.jsonl`` from ``reclose_eval.py``). Along each
fixed 10.05 trajectory, from the first reasoning token through the end of its scored text, every position is
treated as an independent chance to emit ``<|end_think|>``; nothing assumes 10.05 closed in the right place.

Dumps written after commit 9893fb5798 carry the exact ``p_ge1`` and ``p_ge2``. Older dumps keep only three
segment summaries: the mean close probability over the reasoning (``p_premature``, ``L`` positions), the
probability at 10.05's close (``p_first_close``), and over the continuation the exact chance of no close
(``1 - hazard``) and the mean (``mean_p``, ``n`` positions). For those, each multi-position segment is treated
as ``Binomial(length, mean)``, which overstates its no-close probability (AM-GM), so ``p_ge1`` is a lower bound
there; the exact-probe run measures the error.

Writes ``OUT_PREFIX.csv`` and ``OUT_PREFIX.png`` and prints the first step where ``p_ge1`` reaches
``CLOSE_LEARNED`` and no-lead-in ``p_ge2`` reaches ``DOUBLE_LEARNED`` for each run and group.
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
DOUBLE_LEARNED = 0.05
GROUPS = {"swe": "SWE", "other+tool": "Tool-using", "other-plain": "Plain"}
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e6e5e0"


def key(row: dict) -> tuple:
    return row["group"], row["id"], row["sample"], row["bridge"]


def segment_lengths(t: dict) -> tuple[int, int]:
    last = t["reclose"] if t["reclose"] is not None else len(t["ids"]) - 1
    return t["close"] - t["reasoning_start"], last - t["close"]


def binomial_none_and_one(length: int, mean: float) -> tuple[float, float]:
    none = (1.0 - mean) ** length
    one = length * mean * (1.0 - mean) ** (length - 1) if length else 0.0
    return none, one


def estimated_tail(row: dict, reasoning: int, continuation: int) -> tuple[float, float]:
    """P(at least one) and P(at least two) closes from the segment summaries of an older dump."""
    r0, r1 = binomial_none_and_one(reasoning, row["p_premature"])
    c = row["p_first_close"]
    k0 = 1.0 - row["hazard"]
    k1 = min(binomial_none_and_one(continuation, row["mean_p"])[1], row["hazard"])
    none = r0 * (1.0 - c) * k0
    one = r1 * (1.0 - c) * k0 + r0 * c * k0 + r0 * (1.0 - c) * k1
    return 1.0 - none, max(0.0, 1.0 - none - one)


def step_cells(path: Path, lengths: dict[tuple, tuple[int, int]], exact: bool) -> dict[str, tuple[float, float]]:
    cells = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            tail = (row["p_ge1"], row["p_ge2"]) if exact else estimated_tail(row, *lengths[key(row)])
            cells[f"{row['group']}/{'lead_in' if row['bridge'] else 'no_lead_in'}"].append(tail)
    return {cell: tuple(np.mean(values, axis=0)) for cell, values in cells.items()}


def first_step(series: dict[int, float], threshold: float) -> int | None:
    return next((s for s in sorted(series) if series[s] >= threshold), None)


def load_run(directory: str, lengths: dict, exact: bool) -> dict[int, dict]:
    return {
        int(p.stem.split("-")[1]) - FIRST_STEP: step_cells(p, lengths, exact)
        for p in sorted(Path(directory).glob("step-*.jsonl"))
    }


def main() -> None:
    with open(sys.argv[1]) as handle:
        lengths = {key(t): segment_lengths(t) for t in map(json.loads, handle)}
    out = sys.argv[2]
    runs = {}
    for spec in sys.argv[3:]:
        name, directory = spec.split("=", 1)
        sample = next(Path(directory).glob("step-*.jsonl"))
        with open(sample) as handle:
            exact = "p_ge1" in json.loads(handle.readline())
        runs[name] = load_run(directory, lengths, exact)
        if exact:
            runs[f"{name} (estimate)"] = load_run(directory, lengths, False)

    with open(f"{out}.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "step", "cell", "p_ge1", "p_ge2"])
        for name, steps in runs.items():
            for step, cells in sorted(steps.items()):
                for cell, (ge1, ge2) in sorted(cells.items()):
                    writer.writerow([name, step, cell, ge1, ge2])

    print(f"k: first step with P(>=1) >= {CLOSE_LEARNED}; j: first step with P(>=2) >= {DOUBLE_LEARNED}; no lead-in")
    for name, steps in runs.items():
        parts = []
        for group in GROUPS:
            ge1 = {s: c[f"{group}/no_lead_in"][0] for s, c in steps.items()}
            ge2 = {s: c[f"{group}/no_lead_in"][1] for s, c in steps.items()}
            parts.append(
                f"{group} k={first_step(ge1, CLOSE_LEARNED)} j={first_step(ge2, DOUBLE_LEARNED)}"
                f" P(>=1)@0={ge1[min(ge1)]:.2f} P(>=2)@0={ge2[min(ge2)]:.3f}"
            )
        print(f"  {name:<24}" + "   ".join(parts) + f"   (last step {max(steps)})")

    rows = (
        ("no_lead_in", 0, "P(closed ≥1 | opened)", "linear", CLOSE_LEARNED),
        ("no_lead_in", 1, "P(closed ≥2 | opened)", "log", DOUBLE_LEARNED),
        ("lead_in", 1, 'P(closed ≥2 | opened),\nforced "Wait,"-style lead-in', "log", DOUBLE_LEARNED),
    )
    plotted = {name: steps for name, steps in runs.items() if not name.endswith("(estimate)")}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK})
    fig, axes = plt.subplots(len(rows), len(GROUPS), figsize=(13, 10), sharex=True, facecolor=SURFACE)
    for col, (group, title) in enumerate(GROUPS.items()):
        for r, (lead, index, label, scale, threshold) in enumerate(rows):
            ax = axes[r][col]
            ax.set_facecolor(SURFACE)
            for (name, steps), color in zip(plotted.items(), COLORS, strict=False):
                xs = sorted(steps)
                ys = [steps[s][f"{group}/{lead}"][index] for s in xs]
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
    fig.legend(handles, labels, loc="upper center", ncol=len(plotted), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("When the replay learns to close thinking, and to close it twice", y=1.035, fontsize=13)
    fig.text(
        0.5,
        -0.01,
        "Teacher-forced on fixed 10.05 trajectories; any position may close. Runs without exact dumps use a"
        f" segment-mean estimate. Dashed lines: {CLOSE_LEARNED} and {DOUBLE_LEARNED}.",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=130, bbox_inches="tight", facecolor=SURFACE)


if __name__ == "__main__":
    main()
