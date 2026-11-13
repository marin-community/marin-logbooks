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
RULE_COLORS = ("#5F6368", "#FF7F0E", "#2CA02C")


def state_label(state):
    trajectory, phase = state.split("-")
    power = "1/8" if trajectory == "eighth" else "1/4"
    return f"{phase.capitalize()} · trained with {power}"


def style():
    """Adapt Kaiyue's plot defaults to readable standalone comparison figures."""
    mpl.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
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
        if extension == "svg":
            path = output / f"{name}.{extension}"
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
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
        ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 4), useMathText=True)
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
        Line2D([], [], color="#555555", marker=m, linestyle="none", label=state_label(s))
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
        figsize=(6.8 if len(states) == 1 else 4.1 * columns, 3.4 * np.ceil(len(states) / columns)),
        squeeze=False,
    )
    fractions = CANONICAL_FRACTIONS
    for ax, state in zip(axes.flat, states, strict=False):
        result = selected[state]
        rows = {
            (r["draw"], r["alpha"], r["fraction"]): r for r in result["rows"] if r["mode"] == mode and r["groups"] == 8
        }
        for alpha, color in COLORS.items():
            values = 1e4 * np.asarray([[rows[d, alpha, f]["loss_change"] for f in fractions] for d in range(4)])
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
            title=state_label(state),
            xlabel="Common length / reference length",
            ylabel="Loss change (×10⁻⁴)",  # noqa: RUF001 - scientific multiplication sign, not a variable
        )
        ax.set_xticks((0, 0.1, 0.3, 0.5, 1), ("0", "0.1", "0.3", "0.5", "1"))
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    for ax in list(axes.flat)[len(states) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    export(fig, output, f"loss-versus-common-length-{mode}")


def adaptive_plot(results, output):
    rules = ("constant", "linear", "quadratic")
    colors = RULE_COLORS
    names = ("Constant length", "Linear rule", "Quadratic rule")
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
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for index, (rule, color, name) in enumerate(zip(rules, colors, names, strict=True)):
        points = [next(r for r in summaries if r["state"] == s and r["rule"] == rule) for s in states]
        y = np.arange(len(states)) + (index - 1) * 0.22
        for yy, point in zip(y, points, strict=True):
            ax.plot(1e4 * np.array([point["lower"], point["upper"]]), [yy, yy], color=color, linewidth=1.5)
        ax.scatter(
            [1e4 * p["mean"] for p in points], y, color=color, marker=("o", "s", "D")[index], s=35, label=name, zorder=3
        )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(len(states)), [state_label(s) for s in states])
    ax.invert_yaxis()
    ax.set_xlabel("Loss: power 1/8 minus power 1/4 (units of 0.0001)\nNegative favors 1/8 · Positive favors 1/4")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False, fontsize=11)
    fig.tight_layout()
    export(fig, output, "power-comparison-adaptive")
    write_csv(output / "paired-adaptive.csv", summaries)


def adaptive_baseline_plot(results, output):
    """Compare actual quadratic-rule losses with constant lengths within each power."""
    summaries = []
    for result in results:
        if result["stage"] != "heldout":
            raise ValueError("Only held-out direct evaluations may enter this figure")
        rows = {(r["rule"], r["draw"], r["alpha"]): r for r in result["rows"]}
        for alpha in COLORS:
            differences = [
                np.asarray(rows["quadratic", d, alpha]["losses"]) - rows["constant", d, alpha]["losses"]
                for d in range(4)
            ]
            summaries.append(
                dict(state=result["state"], alpha=alpha, **paired_interval(differences, result["reference_weights"]))
            )
    states = [s for s in STATE_ORDER if any(r["state"] == s for r in summaries)]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for index, (alpha, color) in enumerate(COLORS.items()):
        points = [next(r for r in summaries if r["state"] == s and r["alpha"] == alpha) for s in states]
        y = np.arange(len(states)) + (index - 0.5) * 0.25
        for yy, point in zip(y, points, strict=True):
            ax.plot(1e4 * np.array([point["lower"], point["upper"]]), [yy, yy], color=color, linewidth=1.5)
        ax.scatter(
            [1e4 * p["mean"] for p in points],
            y,
            color=color,
            marker=("o", "s")[index],
            s=38,
            label=LABELS[alpha],
            zorder=3,
        )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(len(states)), [state_label(s) for s in states])
    ax.invert_yaxis()
    ax.set_xlabel(
        "Loss: quadratic rule minus constant length (units of 0.0001)\n"
        "Negative favors quadratic · Positive favors constant"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=11)
    fig.tight_layout()
    export(fig, output, "quadratic-versus-constant")
    write_csv(output / "quadratic-versus-constant.csv", summaries)


def envelope_validation_plot(analysis, output):
    """Show each held-out checkpoint rather than concealing failures in an average."""
    fits = analysis["fits"]
    states = [s for s in STATE_ORDER if s in fits]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True)
    for index, (rule, color, marker) in enumerate(
        zip(("constant", "linear", "quadratic"), RULE_COLORS, ("o", "s", "D"), strict=True)
    ):
        y = np.arange(len(states)) + (index - 1) * 0.2
        for ax, metric in zip(axes, ("heldout_pinball", "heldout_coverage"), strict=True):
            ax.scatter(
                [fits[s][rule][metric] for s in states],
                y,
                color=color,
                marker=marker,
                s=40,
                label=f"{rule.capitalize()} envelope",
                zorder=3,
            )
    axes[0].set_yticks(np.arange(len(states)), [state_label(s) for s in states])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="90th-quantile prediction loss", title="Prediction error · lower is better", xlim=(0, None))
    axes[1].set(xlabel="Fraction below predicted envelope", title="Coverage · target is 90%", xlim=(0, 1.05))
    axes[1].set_xticks((0, 0.5, 1), ("0%", "50%", "100%"))
    axes[1].axvline(0.9, color="#555555", linestyle="--", linewidth=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2)
    export(fig, output, "heldout-envelope-validation")


def report_focus_plots(results, analysis, output):
    """Larger single-panel views for the reading guide, retaining full figures too."""
    example = output / "eighth-middle"
    example.mkdir(exist_ok=True)
    common_length_plot([r for r in results if r["state"] == "eighth-middle"], example)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    handles = [
        Line2D(
            [], [], color=c, marker="o", markerfacecolor="none" if a == 0.125 else c, linestyle="none", label=LABELS[a]
        )
        for a, c in COLORS.items()
    ]
    by_state = {r["state"]: r for r in results}
    for state, marker in zip(STATE_ORDER, MARKERS, strict=True):
        rows = [
            r
            for r in by_state[state]["rows"]
            if r["mode"] == "full" and r["groups"] == 8 and r["fraction"] in CANONICAL_FRACTIONS
        ]
        for alpha, color in COLORS.items():
            points = [r for r in rows if r["alpha"] == alpha]
            ax.scatter(
                [r["G"] for r in points],
                [r["Ldir"] for r in points],
                marker=marker,
                s=45 if alpha == 0.125 else 20,
                facecolors="none" if alpha == 0.125 else color,
                edgecolors=color,
                linewidths=0.9,
                alpha=0.75,
            )
        handles.append(Line2D([], [], color="#555555", marker=marker, linestyle="none", label=state_label(state)))
    ax.set(
        xlabel="Reference gradient norm G",
        ylabel="Hessian-product norm ‖Hd‖",
        title="Local curvature along each update direction",
    )
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0.28, 1, 1))
    export(fig, output, "curvature-versus-gradient-focus")
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for index, (rule, color, marker) in enumerate(
        zip(("constant", "linear", "quadratic"), RULE_COLORS, ("o", "s", "D"), strict=True)
    ):
        ax.scatter(
            [analysis["fits"][s][rule]["heldout_coverage"] for s in STATE_ORDER],
            np.arange(6) + (index - 1) * 0.2,
            color=color,
            marker=marker,
            s=42,
            label=rule.capitalize(),
            zorder=3,
        )
    ax.set_yticks(np.arange(6), [state_label(s) for s in STATE_ORDER])
    ax.invert_yaxis()
    ax.axvline(0.9, color="#555555", linestyle="--", linewidth=1)
    ax.set(xlim=(0, 1.05), xlabel="Fraction below predicted envelope", title="Held-out coverage")
    ax.set_xticks((0, 0.5, 1), ("0%", "50%", "100%"))
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    export(fig, output, "heldout-coverage-focus")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--adaptive", type=Path, nargs="*")
    parser.add_argument("--analysis", type=Path)
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
        adaptive_results = [json.loads(p.read_text()) for p in args.adaptive]
        adaptive_plot(adaptive_results, args.output)
        adaptive_baseline_plot(adaptive_results, args.output)
    if args.analysis:
        analysis = json.loads(args.analysis.read_text())
        envelope_validation_plot(analysis, args.output)
        if len(results) == 6:
            report_focus_plots(results, analysis, args.output)


if __name__ == "__main__":
    main()
