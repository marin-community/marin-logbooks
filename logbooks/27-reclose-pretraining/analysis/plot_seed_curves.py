# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot first-close and re-close probe curves for two close-row seeds from ``probe_compare_runs.py`` output.

Usage: plot_seed_curves.py IN_CSV OUT_PNG
"""

import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {"answer": ('" answer" seed (control)', "#2a78d6"), "ltslash": ('"</" seed', "#eb6834")}
GROUPS = {"swe": "SWE", "other+tool": "Tool-using", "other-plain": "Plain"}
ROWS = (
    ("p_first_close", "p(close) at the first close"),
    ("p_reclose", "p(close) at 10.05's re-close"),
    ("ratio", "re-close / first close"),
)
MAX_STEP = 30
RATIO_FROM_STEP = 8
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e6e5e0"


def main() -> None:
    series = defaultdict(list)
    with open(sys.argv[1]) as handle:
        for row in csv.DictReader(handle):
            if row["cell"] in GROUPS and int(row["step"]) <= MAX_STEP:
                series[(row["run"], row["cell"])].append(
                    (int(row["step"]), float(row["p_first_close"]), float(row["p_reclose"]))
                )
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK})
    fig, axes = plt.subplots(len(ROWS), len(GROUPS), figsize=(12, 9.5), sharex=True, facecolor=SURFACE)
    for col, (group, title) in enumerate(GROUPS.items()):
        for r, (metric, label) in enumerate(ROWS):
            ax = axes[r][col]
            ax.set_facecolor(SURFACE)
            for run, (name, color) in RUNS.items():
                points = sorted(series[(run, group)])
                if metric == "ratio":
                    points = [p for p in points if p[0] >= RATIO_FROM_STEP]
                    ys = [p[2] / p[1] for p in points]
                else:
                    ys = [p[1] if metric == "p_first_close" else p[2] for p in points]
                ax.plot([p[0] for p in points], ys, color=color, linewidth=2, label=name)
                ax.plot(
                    points[-1][0], ys[-1], "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2
                )
            if metric == "ratio":
                ax.set_yscale("log")
                ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
            else:
                ax.set_ylim(0, 0.9)
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.tick_params(colors=MUTED)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=12, color=INK, loc="left")
            if col == 0:
                ax.set_ylabel(label, color=INK)
            if r == len(ROWS) - 1:
                ax.set_xlabel("SFT replay step")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(
        'Seeding <|end_think|> from "</" speeds the first close without speeding the re-close',
        y=1.035,
        fontsize=13,
        color=INK,
    )
    fig.text(
        0.5,
        -0.01,
        'Teacher-forced probe on fixed 10.05 trajectories. "</" run stopped at step 20. Ratio row starts at step 8.',
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(sys.argv[2], dpi=130, bbox_inches="tight", facecolor=SURFACE)


if __name__ == "__main__":
    main()
