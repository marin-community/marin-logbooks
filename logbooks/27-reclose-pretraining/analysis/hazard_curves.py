# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""When a replay learns to close thinking, and to close it again, per step.

Usage: hazard_curves.py TRAJECTORIES_JSONL OUT_PREFIX NAME=DIR [NAME=DIR ...] [ref:NAME=PATH ...]

A ``ref:`` entry is a finished checkpoint drawn as a horizontal reference: a directory with one probe dump, or
a ``reclose_logprobs.py`` JSON whose ``trajectories`` follow TRAJECTORIES_JSONL's order. That vLLM path reads
p(close) from the top-20 prompt logprobs, so its values are lower bounds; on 10.05 they match the in-training
probe to within 0.004 on ``p_ge1`` and 0.001 on ``hazard``.

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

Writes ``OUT_PREFIX.csv`` and ``OUT_PREFIX.png`` and prints, for each run and group, the first step where each
metric reaches each reference checkpoint's level.
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
METRICS = ("p_ge1", "hazard")
GROUPS = {"swe": "SWE", "other+tool": "Tool-using", "other-plain": "Plain"}
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
REF_STYLES = ((0, (6, 3)), (0, (1, 2)))
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e6e5e0"


def key(row: dict) -> tuple:
    return row["group"], row["id"], row["sample"], row["bridge"]


def estimated_p_ge1(row: dict, reasoning_length: int) -> float:
    return 1.0 - (1.0 - row["p_premature"]) ** reasoning_length * (1.0 - row["p_first_close"]) * (1.0 - row["hazard"])


def group_means(rows: list[dict], lengths: dict[tuple, int]) -> dict[str, dict[str, float]]:
    cells = defaultdict(list)
    for row in rows:
        if row["bridge"]:
            continue
        p_ge1 = row["p_ge1"] if "p_ge1" in row else estimated_p_ge1(row, lengths[key(row)])
        cells[row["group"]].append((p_ge1, row["hazard"]))
    return {group: dict(zip(METRICS, np.mean(values, axis=0), strict=True)) for group, values in cells.items()}


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def reference_means(path: str, trajectories: list[dict], lengths: dict[tuple, int]) -> dict[str, dict[str, float]]:
    if Path(path).is_dir():
        (dump,) = Path(path).glob("step-*.jsonl")
        return group_means(read_jsonl(dump), lengths)
    with open(path) as handle:
        stats = json.load(handle)["trajectories"]
    fields = ("group", "id", "sample", "bridge")
    rows = [{**{f: t[f] for f in fields}, **s} for t, s in zip(trajectories, stats, strict=True)]
    return group_means(rows, lengths)


def first_step(series: dict[int, float], threshold: float) -> int | None:
    return next((s for s in sorted(series) if series[s] >= threshold), None)


def main() -> None:
    trajectories = read_jsonl(Path(sys.argv[1]))
    lengths = {key(t): t["close"] - t["reasoning_start"] for t in trajectories}
    out = sys.argv[2]
    runs, refs = {}, {}
    for spec in sys.argv[3:]:
        name, path = spec.split("=", 1)
        if name.startswith("ref:"):
            refs[name.removeprefix("ref:")] = reference_means(path, trajectories, lengths)
            continue
        runs[name] = {
            int(p.stem.split("-")[1]) - FIRST_STEP: group_means(read_jsonl(p), lengths)
            for p in sorted(Path(path).glob("step-*.jsonl"))
        }

    with open(f"{out}.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "step", "group", *METRICS])
        for name, steps in runs.items():
            for step, groups in sorted(steps.items()):
                for group, m in sorted(groups.items()):
                    writer.writerow([name, step, group, *(m[k] for k in METRICS)])

    for ref, levels in refs.items():
        print(
            f"reference {ref}: " + "  ".join(f"{g} {levels[g]['p_ge1']:.3f}/{levels[g]['hazard']:.4f}" for g in GROUPS)
        )
    print("first step reaching each reference level, P(>=1 | opened) then hazard")
    for name, steps in runs.items():
        for group in GROUPS:
            series = {k: {s: g[group][k] for s, g in steps.items()} for k in METRICS}
            reached = [
                f"{metric} {ref}={first_step(series[metric], levels[group][metric])}"
                for metric in METRICS
                for ref, levels in refs.items()
            ]
            last = max(steps)
            print(
                f"  {name:<14}{group:<12}"
                + "  ".join(reached)
                + f"  | step {last}: {series['p_ge1'][last]:.3f}/{series['hazard'][last]:.4f}"
            )

    rows = (
        ("p_ge1", "P(closed ≥1 | opened)", "linear"),
        ("hazard", "P(closes again |\na close is in the text)", "log"),
    )
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK})
    fig, axes = plt.subplots(len(rows), len(GROUPS), figsize=(13, 7), sharex=True, facecolor=SURFACE)
    for col, (group, title) in enumerate(GROUPS.items()):
        for r, (metric, label, scale) in enumerate(rows):
            ax = axes[r][col]
            ax.set_facecolor(SURFACE)
            for (name, steps), color in zip(runs.items(), COLORS, strict=False):
                xs = sorted(steps)
                ys = [steps[s][group][metric] for s in xs]
                ax.plot(xs, ys, color=color, linewidth=2, label=name)
                ax.plot(xs[-1], ys[-1], "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
            for (ref, levels), style in zip(refs.items(), REF_STYLES, strict=False):
                level = levels[group][metric]
                ax.axhline(level, color=MUTED, linewidth=1.2, linestyle=style)
                ax.annotate(
                    ref,
                    (1.0, level),
                    xycoords=("axes fraction", "data"),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color=MUTED,
                )
            ax.set_yscale(scale)
            ax.set_ylim(*((0, 1.02) if scale == "linear" else (1e-5, 1)))
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
        " close after 10.05's close token. Gray lines: finished checkpoints.",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=130, bbox_inches="tight", facecolor=SURFACE)


if __name__ == "__main__":
    main()
