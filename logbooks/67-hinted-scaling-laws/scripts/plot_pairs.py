"""Plot the Qwen (benchmark, family) reproductions of Hinted Scaling Laws with marin.scaling_laws.hinted_fit.

Usage: python plot_pairs.py <hinted-scaling-laws checkout> <out dir>
Mirrors experiments/forecasting/plot.py at c0490e53: fit on the t smallest in-forecast models, forecast the largest
at H = 0, t in 2..n-1 on AIME/HLE. Writes three combined figures (one row per pair).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import reproduce_paper as rp
from marin.scaling_laws.hinted_fit import ForecastKind, HintMethod, fit_joint_law, fit_ladder

OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
PAIRS = [("aime", "qwen25", "Qwen2.5"), ("aime", "qwen3", "Qwen3"), ("hle", "qwen25", "Qwen2.5"),
         ("hle", "qwen3", "Qwen3"), ("hle", "qwen35", "Qwen3.5")]
BENCH = {"aime": "AIME", "hle": "HLE"}
FORECAST_YMAX = {("hle", "qwen3"): 0.4, ("hle", "qwen35"): 0.6}
# Printed by `plot.py forecasts --pairs ...` at c0490e53: (pair, t) -> (baseline AE, ensemble AE).
PAPER = {
    ("aime/qwen25", 2): (0.8815, 0.0295), ("aime/qwen25", 3): (0.6684, 0.0798), ("aime/qwen25", 4): (0.2482, 0.0574),
    ("aime/qwen25", 5): (0.1177, 0.0291), ("aime/qwen3", 2): (0.2195, 0.1240), ("aime/qwen3", 3): (0.2156, 0.1704),
    ("aime/qwen3", 4): (0.1705, 0.1173), ("aime/qwen3", 5): (0.1348, 0.0905), ("hle/qwen25", 2): (0.0799, 0.0525),
    ("hle/qwen25", 3): (0.0273, 0.0617), ("hle/qwen25", 4): (0.0109, 0.0456), ("hle/qwen25", 5): (0.0076, 0.0410),
    ("hle/qwen3", 2): (0.0801, 0.0181), ("hle/qwen3", 3): (0.0343, 0.0018), ("hle/qwen3", 4): (0.0590, 0.0244),
    ("hle/qwen3", 5): (0.0635, 0.0274), ("hle/qwen35", 2): (0.2865, 0.0938), ("hle/qwen35", 3): (0.2402, 0.0433),
    ("hle/qwen35", 4): (0.0858, 0.1540),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
SERIES = {
    ForecastKind.MASKING: "#2a78d6",
    ForecastKind.TRUNCATION: "#eb6834",
    ForecastKind.HINTED_ENSEMBLE: "#1baf7a",
    ForecastKind.SIGMOID_BASELINE: "#6b6a64",
}
LABEL = {
    ForecastKind.MASKING: "Masking law, H=0",
    ForecastKind.TRUNCATION: "Truncation law, H=0",
    ForecastKind.HINTED_ENSEMBLE: "Hinted ensemble",
    ForecastKind.SIGMOID_BASELINE: "Sigmoid baseline",
}
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "figure.dpi": 150, "savefig.bbox": "tight",
})


def short(name: str) -> str:
    return name.replace("-Instruct", "").replace("Qwen", "Q")


meta = pd.read_csv(rp.RESULTS / "models.csv")
ladders, names = {}, {}
for bench, fam, _ in PAIRS:
    ladder = rp.pair_ladder(bench, fam)
    m = meta[(meta.benchmark == bench) & (meta.family == fam) & meta.in_forecasts].set_index("model")["capability"]
    ladders[(bench, fam)] = ladder
    names[(bench, fam)] = [m.index[np.isclose(m.to_numpy(), c)][0] for c in ladder.unhinted.capability]
cap_kind = meta.groupby(["benchmark", "family"]).capability_kind.first()

# 1. Hint curves with the all-model joint fit.
fig, axes = plt.subplots(len(PAIRS), 2, figsize=(11, 3.3 * len(PAIRS)), sharey="row")
hgrid = np.linspace(0, 1, 101)
for r, (bench, fam, fam_name) in enumerate(PAIRS):
    ladder, nm = ladders[(bench, fam)], names[(bench, fam)]
    caps = ladder.unhinted.capability
    shades = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, len(caps)))
    for ax, method in zip(axes[r], (HintMethod.TRUNCATION, HintMethod.MASKING)):
        hc = ladder.hinted[method]
        c, h, y = (np.r_[getattr(hc, k), getattr(ladder.unhinted, k)] for k in ("capability", "hint", "accuracy"))
        p = fit_joint_law(c, h, y)
        for i, cap in enumerate(caps):
            sel = np.isclose(c, cap)
            o = np.argsort(h[sel])
            ax.plot(hgrid, p.predict(cap, hgrid), color=shades[i], lw=1.4)
            ax.plot(h[sel][o], y[sel][o], "o", color=shades[i], ms=4, mec="white", mew=0.6, label=short(nm[i]))
        ax.set_title(f"{BENCH[bench]} × {fam_name}, {method.value}\nb1={p.capability_slope:.2f} b2={p.hint_slope:.2f} "
                     f"b0={p.intercept:.2f} A={p.upper:.3f} L={p.lower:.3f} F={p.floor:.4f}", fontsize=8, loc="left")
        ax.set_xlim(-0.02, 1.02)
    axes[r, 0].set_ylabel("Accuracy")
    axes[r, 1].legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2)
for ax in axes[-1]:
    ax.set_xlabel("Hint strength H")
fig.suptitle("Paper's observed accuracy (dots, first 10 rollouts) and our joint-law fit on all in-forecast models (lines)",
             x=0.01, ha="left", fontsize=11)
fig.savefig(OUT / "qwen_hint_curves.png")
plt.close(fig)

# 2 and 3. Forecasts of the largest model for every train size t in 2..n-1.
summary = []
fig, axes = plt.subplots(len(PAIRS), 1, figsize=(8, 3.6 * len(PAIRS)))
fig_e, axes_e = plt.subplots(len(PAIRS), 1, figsize=(7, 3.0 * len(PAIRS)))
for r, (bench, fam, fam_name) in enumerate(PAIRS):
    ladder, nm = ladders[(bench, fam)], names[(bench, fam)]
    caps, acc = ladder.unhinted.capability, ladder.unhinted.accuracy
    target, observed = caps[-1], acc[-1]
    ts = list(range(2, len(caps)))
    errs = {k: [] for k in ForecastKind if k is not ForecastKind.TWO_STAGE_BASELINE}
    for t in ts:
        fc = fit_ladder(ladder.up_to(caps[t - 1])).forecasts(target)
        for k in errs:
            errs[k].append(abs(fc[k] - observed))
        summary.append({"pair": f"{bench}/{fam}", "t": t, "observed": float(observed),
                        **{k.value: float(fc[k]) for k in errs},
                        "paper_baseline_ae": PAPER[(f"{bench}/{fam}", t)][0],
                        "paper_ensemble_ae": PAPER[(f"{bench}/{fam}", t)][1]})

    # Headline panel: the largest train size (n-1 fitted, only the target held out).
    t = ts[-1]
    fit = fit_ladder(ladder.up_to(caps[t - 1]))
    cgrid = np.linspace(caps[0] - 0.2, caps[-1] + 0.35, 200)
    curves = {
        ForecastKind.MASKING: fit.joint[HintMethod.MASKING].predict(cgrid, 0.0),
        ForecastKind.TRUNCATION: fit.joint[HintMethod.TRUNCATION].predict(cgrid, 0.0),
        ForecastKind.SIGMOID_BASELINE: fit.sigmoid_baseline.predict(cgrid),
    }
    curves[ForecastKind.HINTED_ENSEMBLE] = (curves[ForecastKind.MASKING] + curves[ForecastKind.TRUNCATION]) / 2
    ax = axes[r]
    for k in (ForecastKind.MASKING, ForecastKind.TRUNCATION, ForecastKind.HINTED_ENSEMBLE, ForecastKind.SIGMOID_BASELINE):
        ax.plot(cgrid, curves[k], color=SERIES[k], lw=1.8, ls="--" if k is ForecastKind.SIGMOID_BASELINE else "-",
                label=LABEL[k])
    ax.plot(caps[:t], acc[:t], "o", color=INK, ms=5, mec="white", label="Unhinted, fitted")
    ax.plot(caps[t:], acc[t:], "o", color="white", ms=6, mec=INK, mew=1.4, label="Unhinted, held out")
    fc = fit.forecasts(target)
    ax.text(0.99, 0.04, f"{short(nm[-1])} observed {observed:.3f}\nmasking {fc[ForecastKind.MASKING]:.3f}, "
            f"truncation {fc[ForecastKind.TRUNCATION]:.3f}\nensemble {fc[ForecastKind.HINTED_ENSEMBLE]:.3f}, "
            f"sigmoid {fc[ForecastKind.SIGMOID_BASELINE]:.3f}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color=INK)
    ax.set_xticks(caps)
    ax.set_xticklabels([short(n) for n in nm], rotation=30, fontsize=7)
    ax.set_ylim(-0.01, FORECAST_YMAX.get((bench, fam), 1.0))
    ax.set_ylabel("Unhinted accuracy")
    ax.set_title(f"{BENCH[bench]} × {fam_name}: forecast {short(nm[-1])} from the {t} smaller models "
                 f"(x = {cap_kind[(bench, fam)]})", loc="left", fontsize=9)
    if r == 0:
        ax.legend(frameon=False, fontsize=7, loc="upper left")

    ax = axes_e[r]
    for k, idx in ((ForecastKind.SIGMOID_BASELINE, 0), (ForecastKind.HINTED_ENSEMBLE, 1)):
        ax.plot(ts, 100 * np.array(errs[k]), "-o", color=SERIES[k], lw=1.8, ms=6, mec="white", label=f"{LABEL[k]} (ours)")
        ax.plot(ts, [100 * PAPER[(f"{bench}/{fam}", tt)][idx] for tt in ts], "x", color=INK, ms=7, mew=1.4,
                label="Paper" if idx == 1 else None)
    for k in (ForecastKind.MASKING, ForecastKind.TRUNCATION):
        ax.plot(ts, 100 * np.array(errs[k]), ":", color=SERIES[k], lw=1.2, label=f"{LABEL[k]} (ours)")
    ax.set_xticks(ts)
    ax.set_xticklabels([f"{tt}\n(to {short(nm[tt - 1])})" for tt in ts], fontsize=7)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("AE (pp)")
    ax.set_title(f"{BENCH[bench]} × {fam_name}: error forecasting {short(nm[-1])}", loc="left", fontsize=9)
    if r == 0:
        ax.legend(frameon=False, fontsize=7)
axes_e[-1].set_xlabel("Models fitted")
fig.savefig(OUT / "qwen_forecast_largest_t.png")
fig_e.savefig(OUT / "qwen_forecast_error.png")
plt.close(fig)
plt.close(fig_e)
(OUT / "qwen_forecasts.json").write_text(json.dumps(summary, indent=1))
for row in summary:
    ours_b = abs(row["sigmoid_baseline"] - row["observed"])
    ours_e = abs(row["hinted_ensemble"] - row["observed"])
    print(f"{row['pair']:12s} t={row['t']} obs {row['observed']:.3f} mask {row['masking']:.3f} trunc {row['truncation']:.3f} "
          f"ens {row['hinted_ensemble']:.3f} sig {row['sigmoid_baseline']:.3f} | AE base {ours_b:.4f} (paper "
          f"{row['paper_baseline_ae']:.4f}) ens {ours_e:.4f} (paper {row['paper_ensemble_ae']:.4f})")
print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
