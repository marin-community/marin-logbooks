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


def envelope_coverage_panel(summary, ax):
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
        ax.scatter(np.arange(4) + offsets[control], values, c=color, s=48, label=names[control], zorder=3)
    ax.axhline(1, color="black", linestyle="--", linewidth=1.3)
    rhos = [r["normalized_hessian_norm"] for r in summary["rows"]]
    if max(rhos) / max(min(rhos), 1e-30) > 20:
        ax.set_yscale("log")
    ax.set(
        xticks=range(4),
        xticklabels=["Query\n1/8 state", "Query\n1/4 state", "Gate\n1/8 state", "Gate\n1/4 state"],
        ylabel="Normalized Hessian norm\nAbove 1: nominal bound violation",
        title="A  Frozen bounds on new directions",
    )
    ax.legend(loc="best", fontsize=10)


def envelope_tightness_panel(summary, ax):
    names = {"oriented": "True gradient axes", "rotated": "Rotated gradient axes"}
    for control, offset in (("oriented", -0.14), ("rotated", 0.14)):
        values = []
        for block in ("attention", "mlp"):
            rows = {r["control"]: r for r in summary["boundary_matched_coverage"] if r["block"] == block}
            values.append(rows[control]["mean_trace"] / rows["scalar"]["mean_trace"])
        ax.scatter(values, np.arange(2) + offset, color=COLORS[control], s=55, label=names[control], zorder=3)
        for index, value in enumerate(values):
            ax.annotate(f"{value:.2f}", (value, index + offset), xytext=(7, 0), textcoords="offset points", va="center")
    ax.axvline(1, color="#5F6368", ls="--", lw=1.3)
    ax.set_xscale("log")
    ax.set(
        yticks=[0, 1],
        yticklabels=["Attention query", "MLP gate"],
        ylim=(1.55, -0.55),
        xlim=(0.12, 17),
        xlabel="Envelope size / scalar size (log scale)\nLeft of 1: smaller at the same maximum test ratio",
        title="B  Post-hoc matching to maximum ratio 1",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), fontsize=10)


def envelope_figure(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), layout="constrained")
    envelope_coverage_panel(summary, axes[0])
    envelope_tightness_panel(summary, axes[1])
    finish(fig, axes, path)
    for name, renderer in (("coverage", envelope_coverage_panel), ("tightness", envelope_tightness_panel)):
        fig, ax = plt.subplots(figsize=(7, 4.9), layout="constrained")
        renderer(summary, ax)
        finish(fig, [ax], path.with_name(path.name + "-" + name))


def envelope_path_figure(summary, path):
    """Show worst sampled path coverage without implying coverage between points."""
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    groups = [(state, block) for block in ("attention", "mlp") for state in ("eighth-middle", "quarter-middle")]
    names = {"scalar": "Scalar", "oriented": "True gradient axes", "rotated": "Rotated gradient axes"}
    for control, offset in (("scalar", -0.24), ("oriented", 0.0), ("rotated", 0.24)):
        for reference, shift, marker in (("starting", -0.04, "o"), ("current", 0.04, "s")):
            values = [
                max(
                    r["normalized_hessian_norm"]
                    for r in summary["rows"]
                    if (r["state"], r["block"], r["control"], r["gradient_reference"]) == (*group, control, reference)
                )
                for group in groups
            ]
            ax.scatter(
                np.arange(4) + offset + shift,
                values,
                color=COLORS[control],
                marker=marker,
                facecolors="none" if reference == "starting" else COLORS[control],
                s=48,
                label=f"{names[control]} · {reference} gradient",
                zorder=3,
            )
    ax.axhline(1, color="black", linestyle="--", linewidth=1.3)
    ax.set(
        yscale="log",
        xticks=range(4),
        xticklabels=["Query\n1/8 state", "Query\n1/4 state", "Gate\n1/8 state", "Gate\n1/4 state"],
        ylabel="Maximum normalized Hessian norm\nAcross five sampled path points; above 1 violates",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9)
    finish(fig, [ax], path)


def muon_figure(summaries, path, rules=("equal", "direct")):
    names = {
        "blend_0.02": "Blend 0.02",
        "blend_0.05": "Blend 0.05",
        "F5_tangent_0.05": "F5 tangent 0.05",
        "hesscorr_0.05": "Hesscorr 0.05",
        "hesscorr_0.3": "Hesscorr 0.3",
    }
    colors = ["#1F77B4", "#8C62AA", "#4C956C"]
    titles = {
        "equal": "Equal physical length",
        "direct": "Grid-calibrated length",
        "quadratic": "Quadratic-model length",
    }
    fig, axes = plt.subplots(1, len(rules), figsize=(6 * len(rules), 4.7), layout="constrained", squeeze=False)
    axes = axes.ravel()
    for panel, (ax, rule) in enumerate(zip(axes, rules, strict=True)):
        title = f"{chr(65 + panel)}  {titles[rule]}"
        maximum = max(
            abs(interval[bound])
            for summary in summaries
            for interval in summary["paired_vs_plain_muon"][rule].values()
            for bound in ("lower", "upper")
        )
        scale = 1e6 if 0 < maximum < 0.001 else 1
        units = " (millionths)" if scale == 1e6 else ""
        if rule == "direct" and all(s["direct_zero_steps"] == s["candidate_count"] for s in summaries):
            ax.set_title(title)
            ax.text(
                0.5,
                0.5,
                "Every calibrated step was zero.\n\nEvaluation loss change = 0\n"
                "for every candidate.\n\nThis grid does not rank\nsmall-step directions.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
                linespacing=1.6,
            )
            ax.set_axis_off()
            continue
        for state_index, summary in enumerate(summaries):
            offset = (state_index - (len(summaries) - 1) / 2) * 0.18
            color = colors[state_index % len(colors)]
            for index, name in enumerate(names):
                interval = summary["paired_vs_plain_muon"][rule][name]
                y = index + offset
                ax.hlines(y, scale * interval["lower"], scale * interval["upper"], color=color, lw=1.6)
                ax.scatter(
                    scale * interval["mean"],
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
            xlabel=f"Loss difference from plain Muon{units}\n← Feedback helps        Feedback hurts →",
        )
        ax.invert_yaxis()
    axes[0].legend(title="Saved checkpoint", loc="lower right", fontsize=10)
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


def muon_validation_figure(before, after, path):
    """Show measured rounding sensitivity without implying optimizer effectiveness."""
    for field in ("state", "counter", "data_manifest_sha256", "calibration_baseline"):
        if before[field] != after[field]:
            raise ValueError(f"Derivative comparison changed {field}")
    fig, ax = plt.subplots(figsize=(8.8, 4.7), layout="constrained")
    checks = before["derivative_checks"][0]["F5_finite_difference"]
    ax.plot(
        [c["epsilon"] for c in checks],
        [100 * c["relative_error"] for c in checks],
        "o--",
        color="#5F6368",
        lw=1.8,
        label="Original precision · update batch 1",
    )
    for index, record in enumerate(after["derivative_checks"]):
        checks = record["F5_finite_difference"]
        if not record["passed"]:
            raise ValueError("FP32 derivative validation remains incomplete")
        ax.plot(
            [c["epsilon"] for c in checks],
            [100 * c["relative_error"] for c in checks],
            "o-",
            color="#1F77B4",
            lw=2,
            alpha=0.7,
            label="FP32 multiplication · four update batches" if index == 0 else None,
        )
    ax.axhline(2, color="#FF7F0E", lw=1.6, ls=":", label="Acceptance threshold · 2%")
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(0.045, 0.00007),
        xlabel="Relative input perturbation (smaller →)",
        ylabel="Derivative relative error (%) · lower is better",
    )
    ax.set_xticks([0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001], ["0.03", "0.01", "0.003", "0.001", "0.0003", "0.0001"])
    ax.set_yticks([0.01, 0.1, 1, 10, 100, 1000, 10000], ["0.01", "0.1", "1", "10", "100", "1,000", "10,000"])
    ax.legend(loc="lower right", fontsize=10)
    finish(fig, ax, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--path-summary", type=Path)
    parser.add_argument("--muon-validation", type=Path, nargs=2, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--power", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.summary is None and args.muon_validation is None and args.path_summary is None:
        parser.error("Provide a validated summary or Muon derivative records")
    summary = json.loads(args.summary.read_text()) if args.summary else {}
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    if args.path_summary:
        envelope_path_figure(json.loads(args.path_summary.read_text()), args.output / "envelope-path-coverage")
    if args.muon_validation:
        muon_validation_figure(
            *(json.loads(p.read_text()) for p in args.muon_validation), args.output / "muon-derivative-validation"
        )
    if summary.get("power"):
        power_figure(
            summary["power"],
            [json.loads((p / "power-results.json").read_text()) for p in args.power],
            args.output / "power-predictions",
        )
    if summary.get("history"):
        history_figure(summary["history"], args.output / "history-effects")
    if summary.get("muon"):
        muon_figure(summary["muon"], args.output / "muon-controls")
        muon_figure(summary["muon"], args.output / "muon-equal-controls", rules=("equal",))
        muon_figure(summary["muon"], args.output / "muon-direct-controls", rules=("direct",))
        muon_figure(summary["muon"], args.output / "muon-quadratic-controls", rules=("quadratic",))
    if "envelope" in summary:
        envelope_figure(summary["envelope"], args.output / "envelope-orientation")


if __name__ == "__main__":
    main()
