"""Scatter our fresh-rollout cell accuracies against the paper's for the Delphi and Qwen replications.

Usage: python plot_agreement.py <delphi_compare.csv> <qwen_compare.csv> <out.png>
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"none": "#6b6a64", "no_hint": "#6b6a64", "truncation": "#eb6834", "truncate": "#eb6834", "masking": "#2a78d6", "mask": "#2a78d6"}
LABELS = {"none": "unhinted", "no_hint": "unhinted", "truncation": "truncation", "truncate": "truncation", "masking": "masking", "mask": "masking"}


def panel(ax, table: pd.DataFrame, title: str) -> None:
    for method, cells in table.groupby("method"):
        ax.scatter(cells["paper"], cells["ours"], s=14, alpha=0.8, color=COLORS[method], label=LABELS[method])
    ax.plot([0, 1], [0, 1], color="#1f1f1e", lw=0.8)
    beyond = (table["z"].abs() > 3).sum()
    ax.set_title(f"{title}\n{len(table)} cells, {beyond} beyond 3σ, median per-problem r = {table['corr'].median():.2f}", fontsize=10)
    ax.set_xlabel("paper accuracy (released rollouts)")
    ax.set_ylabel("our accuracy (fresh rollouts)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color="#e4e3de", lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def main() -> None:
    delphi, qwen = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    panel(axes[0], delphi, "Delphi GSM8K, 12 rungs (original seed)")
    panel(axes[1], qwen, "AIME, Qwen2.5-1.5B and 3B-Instruct")
    fig.tight_layout()
    fig.savefig(sys.argv[3], dpi=150)


if __name__ == "__main__":
    main()
