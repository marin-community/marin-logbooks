# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Logbook figures for the learning-step and unlikelihood entries, from the page's bundled series.

Writes ``learning_steps.png`` (trough-to-peak normalized close vs close again, per start and turn group) and
``unlikelihood.png`` (no penalty vs weight 10 vs weight 100: re-close rate, closing, and re-close at equal
closing).
"""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "page_data.json").read_text())

INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#101317", "#4b5260", "#858b96", "#e3e6ea", "#c2c7cf", "#ffffff"
CLOSE, AGAIN, REF = "#6250d6", "#e34948", "#6e7480"
WEIGHTS = {"none": ("no penalty", "#86b6ef"), "w10": ("weight 10", "#2a78d6"), "w100": ("weight 100", "#104281")}
GROUPS = {"swe": "SWE", "tool": "Tool-using", "plain": "Plain"}
NORM_PANELS = {
    "answer_point": '" answer" start\nat 10.05\'s close / re-close points',
    "mean_point": "Mean-row start\nat 10.05's close / re-close points",
    "mean_any": "Mean-row start\nP(closes ≥1) / P(closes ≥2)",
}

plt.rcParams.update(
    {
        "font.family": ["IBM Plex Sans", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 9.5,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
    }
)


def end_labels(ax, ends):
    """Label each series at its last point, spreading labels so they do not overlap."""
    low, high = ax.get_ylim()
    gap = 0.06 * (high - low)
    placed = []
    for x, y, text, color in sorted(ends, key=lambda e: e[1]):
        ax.plot(x, y, "o", ms=4, color=color, mec=SURFACE, mew=1.2, zorder=4)
        label_y = max(y, placed[-1] + gap) if placed else y
        placed.append(label_y)
        ax.annotate(
            text,
            (x, y),
            xytext=(x, label_y),
            textcoords="data",
            va="center",
            fontsize=8,
            color=INK_2,
            annotation_clip=False,
        )
        ax.texts[-1].set_x(x)
        ax.texts[-1].set_position((x + 0.012 * (ax.get_xlim()[1] - ax.get_xlim()[0]), label_y))


def learning_steps_figure() -> None:
    fig, axes = plt.subplots(3, 3, figsize=(10, 8.2), sharex=True, sharey=True)
    for col, (key, title) in enumerate(NORM_PANELS.items()):
        for row, (group, label) in enumerate(GROUPS.items()):
            ax = axes[row, col]
            series = DATA["norm"][key][group]
            ax.axhline(0.5, color=REF, lw=1, ls=(0, (4, 4)), zorder=1)
            for name, color, half in (("close", CLOSE, "c50"), ("reclose", AGAIN, "r50")):
                xs, ys = zip(*series[name], strict=True)
                ax.plot(xs, ys, color=color, zorder=3)
                ax.axvline(series[half], ymin=0.46, ymax=0.54, color=color, lw=2, zorder=3)
            ax.text(
                *((0.97, 0.06) if key == "answer_point" else (0.03, 0.9)),
                f"half learned: close {series['c50']}, again {series['r50']}",
                transform=ax.transAxes,
                fontsize=8,
                color=INK_2,
                ha="right" if key == "answer_point" else "left",
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5},
            )
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.03, 1.08)
            if row == 0:
                ax.set_title(title, loc="left")
            if col == 0:
                ax.set_ylabel(f"{label}\nshare of rise")
            if row == 2:
                ax.set_xlabel("training step")
    handles = [
        plt.Line2D([], [], color=CLOSE, label="close"),
        plt.Line2D([], [], color=AGAIN, label="close again"),
        plt.Line2D([], [], color=REF, ls=(0, (4, 4)), lw=1, label="half learned"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle(
        "Closing and closing twice are learned in the same steps",
        x=0.01,
        y=0.995,
        ha="left",
        fontweight="semibold",
        fontsize=12,
    )
    fig.text(
        0.01,
        0.945,
        "Each curve rescaled from its trough (0) to its peak (1) over its largest rise; one replay per start.",
        fontsize=8.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(HERE / "learning_steps.png", dpi=170)
    plt.close(fig)


def unlikelihood_figure() -> None:
    fig, axes = plt.subplots(3, 3, figsize=(10, 8.2))
    titles = ("Re-close rate by step", "Closes at least once by step", "Re-close rate at equal closing")
    for row, (group, label) in enumerate(GROUPS.items()):
        ends = {0: [], 1: [], 2: []}
        for key, (name, color) in WEIGHTS.items():
            pts = DATA["inloop"][key][group]
            steps = [p[0] for p in pts]
            p_ge1 = [p[1] for p in pts]
            hazard = [p[2] for p in pts]
            for col, xs, ys in ((0, steps, hazard), (1, steps, p_ge1), (2, p_ge1, hazard)):
                axes[row, col].plot(xs, ys, color=color, label=name)
                ends[col].append((xs[-1], ys[-1], name.replace("weight ", "W "), color))
        axes[row, 0].set_ylabel(f"{label}\nprobability")
        axes[row, 1].set_ylim(0, 1)
        axes[row, 2].set_xlim(0, 1.0)
        for ax in axes[row, :2]:
            ax.set_xlim(0, 36)
            ax.set_xticks(range(0, 31, 10))
        for col in range(3):
            if row == 0:
                axes[row, col].set_title(titles[col], loc="left")
            if row == 2:
                axes[row, col].set_xlabel("P(closed ≥1)" if col == 2 else "training step")
            axes[row, col].set_ylim(bottom=0)
        for col in range(3):
            end_labels(axes[row, col], ends[col])
    handles = [plt.Line2D([], [], color=c, label=n) for n, c in WEIGHTS.values()]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle(
        "Unlikelihood on a re-close: weight 100 halves it at equal closing",
        x=0.01,
        y=0.995,
        ha="left",
        fontweight="semibold",
        fontsize=12,
    )
    fig.text(
        0.01,
        0.945,
        'Training-loop probe on the " answer" replay, 30 steps; one run per weight.',
        fontsize=8.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(HERE / "unlikelihood.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    learning_steps_figure()
    unlikelihood_figure()
