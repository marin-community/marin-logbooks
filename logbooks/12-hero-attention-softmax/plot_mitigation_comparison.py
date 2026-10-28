# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot gradient accuracy from the public frozen-checkpoint comparison summary."""

import argparse
import gzip
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter

VARIANTS = (
    ("baseline", "Baseline", "#3B6FB6", "o", "-"),
    ("fp32_saved_output", "FP32 saved O", "#D98600", "o", "-"),
    ("fp32_saved_output_shadow", "Fixed-forward shadow", "#965A00", "D", "--"),
    ("native_gproj_hybrid", "GProj prototype", "#188A6B", "s", "-"),
)


def _plot_series(axis, x, values, variant):
    _, label, color, marker, linestyle = variant
    shadow = variant[0] == "fp32_saved_output_shadow"
    axis.plot(
        x,
        values,
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.35 if shadow else 2,
        markersize=7 if shadow else 4.5,
        markerfacecolor="none" if shadow else color,
        markeredgewidth=1.2,
        zorder=4 if shadow else 3,
    )


def _decorate(figure, axes, *, title, subtitle, footer):
    figure.suptitle(title, x=0.5, y=0.985, fontsize=16, fontweight="semibold")
    figure.text(0.5, 0.92, subtitle, ha="center", va="center", fontsize=9.5, color="#444444")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.865), ncol=4, frameon=False)
    figure.text(0.085, 0.035, footer, ha="left", va="bottom", fontsize=8.5, color="#444444", linespacing=1.5)
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)


def _save(figure, destination, data, *, dpi):
    figure.savefig(
        destination,
        dpi=dpi,
        facecolor="white",
        metadata={
            "Title": "Hero frozen attention gradient mitigation comparison",
            "Description": "Relative L2 gradient error versus stable CPU FP64 on the same quantized inputs.",
            "Source": data["source_commit_url"],
        },
    )
    plt.close(figure)


def plot_checkpoints(data, output_dir, *, dpi):
    steps = data["method"]["checkpoint_steps"]
    rows = {(row["step"], row["variant"]): row for row in data["checkpoint_error_rows"]}
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), sharey=True)
    maximum = 0
    for axis, gradient in zip(axes, ("dq", "dk"), strict=True):
        for variant in VARIANTS:
            values = [rows[(step, variant[0])][f"{gradient}_relative_percent"] for step in steps]
            if not all(math.isfinite(value) and value >= 0 for value in values):
                raise ValueError("Checkpoint plot requires finite nonnegative relative errors")
            maximum = max(maximum, *values)
            _plot_series(axis, steps, values, variant)
        axis.set_title(f"${gradient[0]}{gradient[1].upper()}$", fontsize=12)
        axis.set_xticks(steps, [f"{step // 1000}k" for step in steps])
        axis.set_xlabel("Frozen checkpoint")
        axis.margins(x=0.13)
    axes[0].set_ylim(0, maximum * 1.14)
    axes[0].set_ylabel("Relative L2 error (%)")
    _decorate(
        figure,
        axes,
        title="Hero attention gradient accuracy",
        subtitle="Stable CPU FP64 reference · identical quantized inputs and selected-query cotangents",
        footer=(
            "Four queries per head, 48 heads, 64 rank panels and 10 sampled layers; pooled by squared norms.\n"
            "FP32 saved O and its fixed-forward shadow nearly overlap. GProj is a sampled accuracy prototype."
        ),
    )
    figure.subplots_adjust(left=0.085, right=0.975, bottom=0.23, top=0.73, wspace=0.12)
    _save(figure, output_dir / "mitigation-checkpoint-qk-errors.png", data, dpi=dpi)


def plot_layers(data, output_dir, *, dpi):
    rows = sorted((row for row in data["layer_rows"] if row["step"] == 330000), key=lambda row: row["layer"])
    layers = [row["layer"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 5.3), sharey=True)
    minimum, maximum = math.inf, 0
    for axis, gradient in zip(axes, ("dq", "dk"), strict=True):
        for variant in VARIANTS:
            values = [100 * row["variants"][variant[0]][gradient]["relative_l2_error"] for row in rows]
            if not all(math.isfinite(value) and value > 0 for value in values):
                raise ValueError("Logarithmic layer plot requires finite positive relative errors")
            minimum, maximum = min(minimum, *values), max(maximum, *values)
            _plot_series(axis, layers, values, variant)
        axis.set_title(f"${gradient[0]}{gradient[1].upper()}$", fontsize=12)
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5]))
        axis.yaxis.set_major_formatter(ScalarFormatter())
        axis.minorticks_off()
        axis.set_xticks(layers, [f"{row['layer']}\n{'G' if row['is_global'] else 'L'}" for row in rows])
        axis.set_xlabel("Layer index (zero-based) · L: local, G: global")
        axis.set_xlim(-2, 49)
        axis.axvspan(14, 16, color="#EEEEEE", zorder=0)
        outlier = next(row for row in rows if row["layer"] == 15)
        baseline = 100 * outlier["variants"]["baseline"][gradient]["relative_l2_error"]
        candidate = 100 * outlier["variants"]["fp32_saved_output"][gradient]["relative_l2_error"]
        axis.annotate(
            f"Layer 15: {baseline:.2f}% → {candidate:.2f}%\n(baseline → FP32 saved O)",
            xy=(15, baseline),
            xytext=(0.43, 0.89),
            textcoords="axes fraction",
            fontsize=8.5,
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.8},
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.93},
        )
    axes[0].set_ylim(minimum * 0.76, maximum * 1.55)
    axes[0].set_ylabel("Relative L2 error (%) · log scale")
    _decorate(
        figure,
        axes,
        title="330k: gradient error by sampled layer",
        subtitle="Same frozen checkpoint and selected-query objective · logarithmic error axis",
        footer=(
            "Only the ten sampled layers are shown; layer 15 is highlighted. FP32 O and shadow nearly overlap.\n"
            "dK uses expanded GQA boundary heads and selected cotangents; "
            "these measurements do not establish model quality."
        ),
    )
    figure.subplots_adjust(left=0.085, right=0.975, bottom=0.26, top=0.74, wspace=0.12)
    _save(figure, output_dir / "mitigation-330k-layer-qk-errors.png", data, dpi=dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    directory = Path(__file__).resolve().parent
    parser.add_argument("--summary", type=Path, default=directory / "mitigation-comparison-summary.json.gz")
    parser.add_argument("--output-dir", type=Path, default=directory)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    with gzip.open(args.summary, "rt") as stream:
        data = json.load(stream)
    if data["status"] != "completed_and_validated":
        raise ValueError("Figures require a completed and validated public summary")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelpad": 8})
    plot_checkpoints(data, args.output_dir, dpi=args.dpi)
    plot_layers(data, args.output_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()
