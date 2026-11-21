"""Fit hinted scaling laws to our fresh Delphi rollouts and to the paper's same-checkpoint rollouts.

Usage: python fit_ours.py <paper data/results dir> <our results dir> <figure out dir>

Both ladders use the checkpoints we sampled (seed "original", or "0" for 5e19 and 6e20). For each number
of fitted rungs (7..11) it forecasts the largest sampled rung at H=0 and prints both sides' forecasts and
errors, then draws hint curves (ours as dots, paper as crosses) with our fitted laws.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from compare_with_paper import our_table, paper_table
from marin.scaling_laws.hinted_fit import ForecastKind, HintMethod, LadderScores, ScoreCells, fit_ladder

METHODS = {HintMethod.TRUNCATION: "truncation", HintMethod.MASKING: "masking"}


def ladder(table: pd.DataFrame, rate: str, capability: pd.Series) -> LadderScores:
    table = table[table["model"].isin(capability.index)].copy()
    table["problem_id"] = table["problem_id"].astype(int)
    wide = table.pivot_table(index=["model", "method", "hint"], columns="problem_id", values=rate)
    wide = wide.dropna()

    def cells(method: str) -> ScoreCells:
        part = wide.xs(method, level="method")
        models = part.index.get_level_values("model")
        return ScoreCells(
            capability=capability.loc[models].to_numpy(dtype=float),
            hint=part.index.get_level_values("hint").to_numpy(dtype=float),
            problem_scores=part.to_numpy(dtype=float),
        )

    unhinted = cells("none")
    order = np.argsort(unhinted.capability)
    unhinted = ScoreCells(unhinted.capability[order], unhinted.hint[order], unhinted.problem_scores[order])
    return LadderScores(unhinted=unhinted, hinted={m: cells(name) for m, name in METHODS.items()}, bits_per_byte=None)


def main() -> None:
    data, ours_dir, fig_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    fig_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(data / "models.csv")
    meta = meta[(meta.benchmark == "gsm8k") & (meta.family == "delphi")]
    ours = our_table(ours_dir)
    complete = ours.groupby("model").size()
    sampled = complete[complete == 256 * 21].index
    capability = meta.set_index("model")["capability"].loc[sorted(sampled, key=lambda m: meta.set_index("model")["capability"][m])]
    paper = paper_table(data / "rollouts")
    ladders = {"ours": ladder(ours, "our_rate", capability), "paper": ladder(paper, "paper_rate", capability)}
    caps = ladders["ours"].unhinted.capability
    target = caps[-1]
    print(f"rungs sampled: {list(capability.index)}")
    print(f"target {capability.index[-1]}: observed ours {ladders['ours'].unhinted.accuracy[-1]:.4f}, "
          f"paper {ladders['paper'].unhinted.accuracy[-1]:.4f}")
    rows = []
    for fitted in range(max(3, len(caps) - 5), len(caps)):
        for side, lad in ladders.items():
            fc = fit_ladder(lad.up_to(caps[fitted - 1])).forecasts(target)
            observed = lad.unhinted.accuracy[-1]
            rows.append({"fitted": fitted, "side": side, **{k.value: v for k, v in fc.items()},
                         "observed": observed,
                         "ensemble_ae": abs(fc[ForecastKind.HINTED_ENSEMBLE] - observed),
                         "sigmoid_ae": abs(fc[ForecastKind.SIGMOID_BASELINE] - observed)})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    table.to_csv(fig_dir / "forecasts_ours_vs_paper.csv", index=False)

    full = {side: fit_ladder(lad) for side, lad in ladders.items()}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(caps)))
    grid = np.linspace(0, 1, 51)
    for ax, (method, name) in zip(axes, METHODS.items()):
        for side, marker in (("ours", "o"), ("paper", "x")):
            lad = ladders[side]
            c = np.r_[lad.unhinted.capability, lad.hinted[method].capability]
            h = np.r_[lad.unhinted.hint, lad.hinted[method].hint]
            y = np.r_[lad.unhinted.accuracy, lad.hinted[method].accuracy]
            for i, cap in enumerate(caps):
                sel = np.isclose(c, cap)
                ax.scatter(h[sel], y[sel], color=colors[i], marker=marker, s=14, alpha=0.8)
        for i, cap in enumerate(caps):
            ax.plot(grid, full["ours"].joint[method].predict(cap, grid), color=colors[i], lw=1,
                    label=capability.index[i].removeprefix("delphi_") if method is HintMethod.TRUNCATION else None)
        p = full["ours"].joint[method]
        ax.set_title(f"{name}: ours (dots, fitted lines) vs paper (x)\n"
                     f"b1={p.capability_slope:.2f} b2={p.hint_slope:.2f} b0={p.intercept:.2f}")
        ax.set_xlabel("hint strength H")
    axes[0].set_ylabel("GSM8K accuracy")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "delphi_ours_vs_paper_hint_curves.png", dpi=150)
    for side, fit in full.items():
        for method, name in METHODS.items():
            p = fit.joint[method]
            print(f"{side:5s} {name:10s} b1={p.capability_slope:.3f} b2={p.hint_slope:.3f} b0={p.intercept:.3f} "
                  f"A={p.upper:.3f} L={p.lower:.3f} F={p.floor:.4f}")


if __name__ == "__main__":
    main()
