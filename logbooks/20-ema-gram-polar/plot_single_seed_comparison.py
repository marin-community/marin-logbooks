# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the adjudicated seed0 comparison from cached, published scalar data."""
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    root = Path(__file__).resolve().parent
    data = json.loads((root / "single-seed-comparison.json").read_text())
    # Adapted from Kaiyue's scientific plotting style; no smoothing or seed bands.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 1.0,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    methods = [
        ("baseline", "0 (completed)", "#5F6368", "-"),
        ("alpha0125", "1/8 (completed)", "#1F77B4", "-"),
        ("alpha025", "1/4 (failed)", "#2C8C69", "--"),
        ("alpha05", "1/2 (failed)", "#9467BD", "--"),
    ]
    handles = []
    for slug, label, color, style in methods:
        rows = data["evaluation_curves"][slug]
        assert len(rows) == 38 and len({r["label"] for r in rows}) == 38
        native = [r for r in rows if r["provenance"] == "W&B"]
        local = [r for r in rows if r["provenance"] != "W&B"]
        axes[0].plot(
            [r["tokens"] / 1e9 for r in native],
            [r["c4_loss"] for r in native],
            color=color,
            ls=style,
            lw=1.7,
            marker="o",
            ms=2.5,
        )
        for r in local:
            axes[0].scatter(
                r["tokens"] / 1e9,
                r["c4_loss"],
                marker="D",
                s=36,
                facecolors="white",
                edgecolors=color,
                linewidths=1.4,
                zorder=4,
            )
        axes[2].plot(
            [r["native_runtime_seconds"] / 3600 for r in native],
            [r["c4_loss"] for r in native],
            color=color,
            ls=style,
            lw=1.7,
            marker="o",
            ms=2.5,
        )
        if slug != "baseline":
            pairs = data["paired_c4_deltas"][slug]
            native_pairs = [r for r in pairs if r["provenance"] == "W&B"]
            axes[1].plot(
                [r["tokens"] / 1e9 for r in native_pairs],
                [r["c4_delta"] for r in native_pairs],
                color=color,
                ls=style,
                lw=1.7,
                marker="o",
                ms=2.5,
            )
            for r in pairs:
                if r["provenance"] != "W&B":
                    axes[1].scatter(
                        r["tokens"] / 1e9,
                        r["c4_delta"],
                        marker="D",
                        s=36,
                        facecolors="white",
                        edgecolors=color,
                        linewidths=1.4,
                        zorder=4,
                    )
        handles.append(Line2D([], [], color=color, ls=style, lw=1.7, label=label))
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(color="#B0B0B0", linestyle="--", linewidth=0.65, alpha=0.5)
        ax.margins(x=0.025, y=0.06)
    for ax in (axes[0], axes[2]):
        ax.axhline(4, color="black", ls=":", lw=1)
        ax.set_ylim(3.68, 6.56)
        ax.set_ylabel("Held-out C4 loss (nats)")
    axes[0].set_xlabel("Training tokens (billions)")
    axes[1].set_xlabel("Training tokens (billions)")
    axes[1].set_ylabel("C4 loss difference from alpha=0")
    axes[1].axhline(0, color="black", ls=":", lw=1)
    axes[2].set_xlabel("Native runtime (hours; queue excluded)")
    axes[0].set_title("(a) Fixed token budget")
    axes[1].set_title("(b) Same-evaluation differences")
    axes[2].set_title("(c) Actual deployment time")
    handles.append(
        Line2D([], [], color="black", marker="D", markerfacecolor="white", ls="none", label="Local final metric")
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1),
        columnspacing=1.3,
        handlelength=2,
    )
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.06, right=0.99, wspace=0.38)
    out = root / "figures"
    out.mkdir(exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(out / f"single-seed-comparison.{extension}", dpi=220, facecolor="white")
    svg = out / "single-seed-comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
