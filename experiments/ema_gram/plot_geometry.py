# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scientific figures from validated checkpoint summaries, with editable vector text."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"scalar": "#5F6368", "oriented": "#1F77B4", "rotated": "#8C62AA"}
POWER_COLORS = {0.125: "#1F77B4", 0.1875: "#4C956C", 0.25: "#8C62AA"}


def style():
    # Adapted from Kaiyue's scientific plotting style; readable at report width.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "axes.linewidth": 1.2,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def finish(fig, axes, path):
    for ax in np.asarray(axes).ravel():
        ax.set_axisbelow(True)
        ax.grid(True, color="#B0B0B0", linestyle="--", linewidth=0.85, alpha=0.5)
    for extension in ("svg", "pdf", "png"):
        fig.savefig(path.with_suffix("." + extension), dpi=220, bbox_inches="tight")
    plt.close(fig)


def power_figure(summary, results, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    values = []
    for alpha, color in POWER_COLORS.items():
        rows = [r["evaluation"]["direct"] for result in results for r in result["rows"] if r["alpha"] == alpha]
        predicted, observed = [-r["predicted_change"] for r in rows], [-r["loss_change"] for r in rows]
        values.extend(predicted + observed)
        axes[0].scatter(
            predicted,
            observed,
            c=color,
            s=28,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.4,
            label={0.125: "1/8", 0.1875: "3/16", 0.25: "1/4"}[alpha],
        )
    low, high = min([*values, 0]), max([*values, 0])
    pad = 0.05 * max(high - low, 1e-8)
    axes[0].plot([low - pad, high + pad], [low - pad, high + pad], color="#5F6368", ls="--", lw=1.2)
    axes[0].set(
        xlim=(low - pad, high + pad),
        ylim=(low - pad, high + pad),
        xlabel="Predicted loss decrease",
        ylabel="Observed evaluation loss decrease",
        title="A  Does the quadratic prediction transfer?",
    )
    axes[0].legend(title="Inverse power", loc="best")
    for i, result in enumerate(summary):
        interval = result["paired_loss_differences"]["direct"]["quadratic_minus_alignment"]
        axes[1].hlines(i, interval["lower"], interval["upper"], color="#1F77B4", lw=1.8)
        axes[1].scatter(interval["mean"], i, color="#1F77B4", s=32)
    axes[1].axvline(0, color="black", lw=1.2)
    axes[1].set(
        yticks=range(len(summary)),
        yticklabels=[r["state"].replace("-", " · ") for r in summary],
        xlabel="Evaluation loss: curvature choice - alignment choice\n← Curvature helps        Curvature hurts →",
        title="B  Does adding curvature improve the choice?",
    )
    axes[1].invert_yaxis()
    finish(fig, axes, path)


def envelope_figure(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    groups = [(state, block) for block in ("attention", "mlp") for state in ("eighth-middle", "quarter-middle")]
    offsets = {"scalar": -0.19, "oriented": 0, "rotated": 0.19}
    names = {"scalar": "Scalar", "oriented": "True gradient axes", "rotated": "Rotated gradient axes"}
    for control, color in COLORS.items():
        values = [
            next(
                r["normalized_hessian_norm"]
                for r in summary["rows"]
                if (r["state"], r["block"], r["control"]) == (*group, control)
            )
            for group in groups
        ]
        axes[0].scatter(np.arange(4) + offsets[control], values, c=color, s=48, label=names[control], zorder=3)
    axes[0].axhline(1, color="black", linestyle="--", linewidth=1.3)
    rhos = [r["normalized_hessian_norm"] for r in summary["rows"]]
    if max(rhos) / max(min(rhos), 1e-30) > 20:
        axes[0].set_yscale("log")
    axes[0].set(
        xticks=range(4),
        xticklabels=["Query\n1/8 state", "Query\n1/4 state", "Gate\n1/8 state", "Gate\n1/4 state"],
        ylabel="Normalized Hessian operator norm\nAbove 1: nominal bound violation",
        title="A  Do frozen bounds cover new directions?",
    )
    axes[0].legend(loc="best", fontsize=10)
    for control in ("oriented", "rotated"):
        values = []
        for block in ("attention", "mlp"):
            rows = {r["control"]: r for r in summary["matched_coverage"] if r["block"] == block}
            values.append(rows[control]["mean_trace"] / rows["scalar"]["mean_trace"])
        offset = -0.17 if control == "oriented" else 0.17
        axes[1].bar(np.arange(2) + offset, values, width=0.3, color=COLORS[control], label=names[control])
    axes[1].axhline(1, color="#5F6368", ls="--", lw=1.3, label="Scalar reference")
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Attention query", "MLP gate"],
        ylim=(0, None),
        ylabel="Envelope size / scalar envelope size\nLower: tighter at matched nominal coverage",
        title="B  Is the matrix envelope more economical?",
    )
    axes[1].legend(loc="best", fontsize=10)
    finish(fig, axes, path)


def muon_figure(summaries, path):
    names = {
        "blend_0.02": "Blend 0.02",
        "blend_0.05": "Blend 0.05",
        "F5_tangent_0.05": "F5 tangent 0.05",
        "hesscorr_0.05": "Hesscorr 0.05",
        "hesscorr_0.3": "Hesscorr 0.3",
    }
    colors = ["#1F77B4", "#8C62AA", "#4C956C"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for ax, rule, title in zip(
        axes, ("equal", "direct"), ("A  Equal physical length", "B  Separately calibrated length"), strict=True
    ):
        for state_index, summary in enumerate(summaries):
            offset = (state_index - (len(summaries) - 1) / 2) * 0.18
            color = colors[state_index % len(colors)]
            for index, name in enumerate(names):
                interval = summary["paired_vs_plain_muon"][rule][name]
                y = index + offset
                ax.hlines(y, interval["lower"], interval["upper"], color=color, lw=1.6)
                ax.scatter(
                    interval["mean"],
                    y,
                    color=color,
                    s=30,
                    label=summary["state"].replace("muon-", "").replace("-", " ") if index == 0 else None,
                )
        ax.axvline(0, color="black", lw=1.2)
        ax.set(
            yticks=range(len(names)),
            yticklabels=list(names.values()),
            title=title,
            xlabel="Evaluation loss difference from plain Muon\n← Feedback helps        Feedback hurts →",
        )
        ax.invert_yaxis()
    axes[1].legend(title="Saved checkpoint", loc="best", fontsize=10)
    finish(fig, axes, path)


def history_figure(summaries, path):
    conditions = summaries[0]["conditions"]
    labels = []
    for condition in conditions:
        if condition["name"] == "saved":
            labels.append("Saved M + saved Grams")
        else:
            source = {"replica0": "A", "replica1": "B", "pooled": "A+B pool"}[condition["replica"]]
            changed = (
                "M + Grams"
                if condition["momentum"] == condition["grams"] == "fresh"
                else ("M only" if condition["momentum"] == "fresh" else "Grams only")
            )
            labels.append(f"Fresh {changed} ({source})")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    for ax, rule, title in zip(
        axes, ("equal", "direct"), ("A  Equal physical length", "B  Separately calibrated length"), strict=True
    ):
        for state_index, summary in enumerate(summaries):
            color = ["#1F77B4", "#8C62AA"][state_index % 2]
            offset = (state_index - (len(summaries) - 1) / 2) * 0.2
            for i, condition in enumerate(conditions):
                interval = summary["quarter_minus_eighth"][rule][condition["name"]]
                ax.hlines(i + offset, interval["lower"], interval["upper"], color=color, lw=1.6)
                ax.scatter(
                    interval["mean"],
                    i + offset,
                    s=28,
                    color=color,
                    label=summary["state"].replace("-", " ") if i == 0 else None,
                )
        ax.axvline(0, color="black", lw=1.2)
        ax.set(
            yticks=range(len(labels)),
            yticklabels=labels,
            title=title,
            xlabel="Evaluation loss: quarter - eighth\n← Quarter better          Eighth better →",
        )
        ax.invert_yaxis()
    axes[1].legend(title="Saved checkpoint", loc="best", fontsize=10)
    finish(fig, axes, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--power", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    if summary["power"]:
        power_figure(
            summary["power"],
            [json.loads((p / "power-results.json").read_text()) for p in args.power],
            args.output / "power-predictions",
        )
    if summary.get("history"):
        history_figure(summary["history"], args.output / "history-effects")
    if summary.get("muon"):
        muon_figure(summary["muon"], args.output / "muon-controls")
    if "envelope" in summary:
        envelope_figure(summary["envelope"], args.output / "envelope-orientation")


if __name__ == "__main__":
    main()
