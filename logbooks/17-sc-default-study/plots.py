# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot evaluated learning checkpoints, complete-cycle cost, and paired intervals."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"tis": "#0072B2", "sc128": "#D55E00", "sc32": "#009E73"}
LABELS = {"tis": "TIS", "sc128": "SC128", "sc32": "SC32"}


def save(figure, output: Path, name: str) -> None:
    for axis in figure.axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
    figure.savefig(output / f"{name}.svg", bbox_inches="tight")
    figure.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot(manifest_path: Path, analysis_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 10})
    analysis = json.loads(analysis_path.read_text())
    runs = [
        row
        for row in json.loads(manifest_path.read_text())["runs"]
        if row["eligible"] and row["seed"] in analysis["seeds"]
    ]
    data = [(row, json.loads((manifest_path.parent / row["collected"] / "summary.json").read_text())) for row in runs]
    if len(data) != 3 * len(analysis["seeds"]):
        raise ValueError("Plot input differs from the complete inference matrix")

    learning, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True, layout="constrained")
    x_fields = ("step", "trained_loss_tokens", "elapsed_including_evaluation_seconds")
    scales = (1, 1e6, 60)
    for arm, color in COLORS.items():
        curves = [
            [
                {
                    **point,
                    "elapsed_including_evaluation_seconds": (
                        summary["cycles"][point["step"] - 1]["elapsed_including_evaluation_seconds"]
                        if point["step"]
                        else 0
                    ),
                }
                for point in summary["curves"]
            ]
            for row, summary in data
            if row["arm"] == arm
        ]
        if any([point["step"] for point in curve] != [0, 25, 50] for curve in curves):
            raise ValueError("Learning curves must retain the fixed checkpoint schedule")
        if any(point["tasks"] != 1000 for curve in curves for point in curve):
            raise ValueError("Every learning-curve point must use the same 1,000-task panel")
        y = np.array([[point["macro_full_success"] * 100 for point in curve] for curve in curves])
        for axis, field, scale in zip(axes, x_fields, scales, strict=True):
            x = np.array([[point[field] / scale for point in curve] for curve in curves])
            for seed_x, seed_y in zip(x, y, strict=True):
                axis.plot(seed_x, seed_y, color=color, alpha=0.3, linewidth=1)
            axis.plot(x.mean(axis=0), y.mean(axis=0), "o-", color=color, label=LABELS[arm], linewidth=2)
    axes[0].set_ylabel("Equal-family full success (%)")
    for axis, label in zip(
        axes,
        (
            "Policy updates",
            "Trained loss tokens (millions)",
            "Elapsed from first rollout (minutes)\nIncludes evaluation pauses",
        ),
        strict=True,
    ):
        axis.set_xlabel(label)
    axes[0].legend(frameon=False)
    learning.suptitle("Learning on the fixed 1,000-task panel; thin lines show matched seeds")
    save(learning, output, "learning-curves")

    cost, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    for arm, color in COLORS.items():
        samples = [summary["cycles"] for row, summary in data if row["arm"] == arm]
        updates = np.array([point["step"] for point in samples[0]])
        for axis, field, scale in zip(
            axes, ("training_cycle_elapsed_seconds", "cumulative_trained_loss_tokens"), (60, 1e6), strict=True
        ):
            values = np.array([[point[field] / scale for point in sample] for sample in samples])
            for seed_y in values:
                axis.plot(updates, seed_y, color=color, alpha=0.3, linewidth=1)
            axis.plot(updates, values.mean(axis=0), color=color, linewidth=2, label=LABELS[arm])
            axis.set_xlabel("Policy updates")
    axes[0].set_ylabel("Training-cycle time (minutes)")
    axes[1].set_ylabel("Trained loss tokens (millions)")
    axes[0].legend(frameon=False)
    cost.suptitle("Complete training cycles; exclusive evaluation sessions excluded")
    save(cost, output, "cost-curves")

    intervals, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    comparisons = list(analysis["comparisons"])
    labels = [name.replace("_versus_", " - ").upper() for name in comparisons]
    for axis, metric, margin in zip(axes, ("quality_difference", "relative_training_cycle_cost"), (-1, 5), strict=True):
        for position, comparison in enumerate(comparisons):
            for confidence, width, color, marker in (
                ("0.975", 1, "#999999", "none"),
                ("0.95", 2.5, "#333333", "o"),
            ):
                value = analysis["comparisons"][comparison]["intervals"][confidence][metric]
                estimate, low, high = (100 * value[key] for key in ("estimate", "low", "high"))
                axis.errorbar(
                    estimate,
                    position,
                    xerr=[[estimate - low], [high - estimate]],
                    fmt=marker,
                    color=color,
                    elinewidth=width,
                    capsize=4,
                )
        axis.axvline(0, color="#999999", linewidth=1)
        axis.axvline(margin, color="#B22222", linestyle="--", linewidth=1)
        axis.set_yticks(range(len(labels)), labels)
        axis.invert_yaxis()
    axes[0].set_xlabel("Held-out success difference (percentage points)")
    axes[1].set_xlabel("Relative training-cycle cost (%)")
    intervals.suptitle(
        "Update-50 endpoint: inner 95%, outer 97.5% intervals\n"
        "Decisions use outer intervals; dashed lines mark margins"
    )
    save(intervals, output, "endpoint-intervals")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.manifest, args.analysis, args.output)
