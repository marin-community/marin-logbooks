# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Readable scientific figures from observed curvature and direct-loss measurements."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.ema_gram.gradient_curvature_analysis import CANONICAL_FRACTIONS, write_csv
from experiments.ema_gram.gradient_curvature_math import paired_interval

COLORS = {0.125: "#1F77B4", 0.25: "#9467BD"}
LABELS = {0.125: "Power 1/8", 0.25: "Power 1/4"}
STATE_ORDER = ("eighth-early", "eighth-middle", "eighth-late", "quarter-early", "quarter-middle", "quarter-late")
MARKERS = ("o", "s", "^", "D", "v", "P")


def style():
    """Adapt Kaiyue's plot defaults to readable standalone comparison figures."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "axes.linewidth": 1.2,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.axisbelow": True,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "grid.color": "#B0B0B0",
            "figure.facecolor": "white",
        }
    )


def export(fig, output, name):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output / f"{name}.{extension}", dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def curvature_plot(results, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    states = {r["state"]: r for r in results}
    for state, marker in zip(STATE_ORDER, MARKERS, strict=True):
        if state not in states:
            continue
        rows = [
            r
            for r in states[state]["rows"]
            if r["mode"] == "full" and r["groups"] == 8 and r["fraction"] in CANONICAL_FRACTIONS
        ]
        for alpha, color in COLORS.items():
            for ax, metric in zip(axes, ("Ldir", "gradient_change_rate"), strict=True):
                points = [r for r in rows if r["alpha"] == alpha and r[metric] is not None]
                ax.scatter(
                    [r["G"] for r in points],
                    [r[metric] for r in points],
                    marker=marker,
                    s=42 if alpha == 0.125 else 18,
                    alpha=0.75,
                    facecolors="none" if alpha == 0.125 else color,
                    edgecolors=color,
                    linewidths=0.9,
                )
    for ax, title, ylabel in zip(
        axes,
        ("Curvature at one position", "Change over a whole step"),
        ("Hessian-product norm  ‖Hd‖", "Gradient-change rate  ‖g(s) - g(0)‖ / s"),
        strict=True,
    ):
        ax.set(xlabel="Reference gradient norm G at this position", ylabel=ylabel, title=title)
        ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 4))
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker="o",
            markerfacecolor="none" if a == 0.125 else color,
            linestyle="none",
            label=LABELS[a],
        )
        for a, color in COLORS.items()
    ]
    handles += [
        Line2D([], [], color="#555555", marker=m, linestyle="none", label=s.replace("-", " "))
        for s, m in zip(STATE_ORDER, MARKERS, strict=True)
        if s in states
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.15, 1, 1), w_pad=2.0)
    export(fig, output, "curvature-versus-gradient")


def common_length_plot(results, output, mode="full"):
    selected = {r["state"]: r for r in results if any(row["mode"] == mode for row in r["rows"])}
    states = [s for s in STATE_ORDER if s in selected]
    columns = min(3, len(states))
    fig, axes = plt.subplots(
        int(np.ceil(len(states) / columns)),
        columns,
        figsize=(4.1 * columns, 3.4 * np.ceil(len(states) / columns)),
        squeeze=False,
    )
    fractions = (0, 0.1, 0.3, 1.0)
    for ax, state in zip(axes.flat, states, strict=False):
        result = selected[state]
        rows = {
            (r["draw"], r["alpha"], r["fraction"]): r for r in result["rows"] if r["mode"] == mode and r["groups"] == 8
        }
        for alpha, color in COLORS.items():
            values = np.asarray([[rows[d, alpha, f]["loss_change"] for f in fractions] for d in range(4)])
            for draw in values:
                ax.plot(fractions, draw, color=color, linewidth=0.7, alpha=0.22)
            ax.plot(
                fractions,
                values.mean(axis=0),
                color=color,
                marker="o" if alpha == 0.125 else "s",
                linestyle="-" if alpha == 0.125 else "--",
                markersize=4,
                linewidth=2.2,
                label=LABELS[alpha],
            )
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set(
            title=state.replace("-", " "),
            xlabel="Common length / reference length",
            ylabel="Loss change (lower is better)",
        )
        ax.set_xticks((0, 0.3, 1))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, 4))
    for ax in list(axes.flat)[len(states) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    export(fig, output, f"loss-versus-common-length-{mode}")


def adaptive_plot(results, output):
    rules = ("constant", "linear", "quadratic")
    colors = ("#5F6368", "#1F77B4", "#9467BD")
    names = ("Constant physical length", "Linear-curvature rule", "Quadratic-curvature rule")
    summaries = []
    for result in results:
        if result["stage"] != "heldout":
            raise ValueError("Only held-out adaptive evaluations may enter this figure")
        for rule in rules:
            rows = {(r["draw"], r["alpha"]): r for r in result["rows"] if r["rule"] == rule}
            differences = [np.asarray(rows[d, 0.125]["losses"]) - rows[d, 0.25]["losses"] for d in range(4)]
            summaries.append(
                dict(state=result["state"], rule=rule, **paired_interval(differences, result["reference_weights"]))
            )
    states = [s for s in STATE_ORDER if any(r["state"] == s for r in summaries)]
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    for index, (rule, color, name) in enumerate(zip(rules, colors, names, strict=True)):
        points = [next(r for r in summaries if r["state"] == s and r["rule"] == rule) for s in states]
        y = np.arange(len(states)) + (index - 1) * 0.22
        for yy, point in zip(y, points, strict=True):
            ax.plot([point["lower"], point["upper"]], [yy, yy], color=color, linewidth=1.5)
        ax.scatter([p["mean"] for p in points], y, color=color, s=35, label=name, zorder=3)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(len(states)), [s.replace("-", " ") for s in states])
    ax.invert_yaxis()
    ax.set_xlabel("Paired loss difference: power 1/8 minus power 1/4\nNegative favors 1/8 · Positive favors 1/4")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    fig.tight_layout()
    export(fig, output, "power-comparison-adaptive")
    write_csv(output / "paired-adaptive.csv", summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--adaptive", type=Path, nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [json.loads(p.read_text()) for p in args.inputs]
    style()
    curvature_plot(results, args.output)
    common_length_plot(results, args.output)
    if any(row["mode"] == "matrix_only" for r in results for row in r["rows"]):
        common_length_plot(results, args.output, "matrix_only")
    if args.adaptive:
        adaptive_plot([json.loads(p.read_text()) for p in args.adaptive], args.output)


if __name__ == "__main__":
    main()
