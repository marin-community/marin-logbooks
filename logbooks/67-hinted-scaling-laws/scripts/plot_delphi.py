"""Plot the Delphi GSM8K reproduction of Hinted Scaling Laws with marin.scaling_laws.hinted_fit."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import reproduce_paper as rp
from marin.scaling_laws.hinted_fit import ForecastKind, HintMethod, fit_joint_law, fit_ladder

OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b", "#0a2c58", "#071f40"]
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
NAMES = ["3e18", "9e18", "2e19", "3e19", "5e19", "9e19", "2e20", "3e20", "6e20", "1e21", "1e22", "1e23"]

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

ladder = rp.pair_ladder("gsm8k", "delphi")
caps = ladder.unhinted.capability
acc = ladder.unhinted.accuracy
assert len(caps) == len(NAMES), caps

# 1. Hint curves with the all-rung joint fit (paper Table 1 setting).
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
hgrid = np.linspace(0, 1, 101)
for ax, method in zip(axes, (HintMethod.TRUNCATION, HintMethod.MASKING)):
    hc = ladder.hinted[method]
    c = np.r_[hc.capability, ladder.unhinted.capability]
    h = np.r_[hc.hint, ladder.unhinted.hint]
    y = np.r_[hc.accuracy, ladder.unhinted.accuracy]
    p = fit_joint_law(c, h, y)
    last_label = -1.0
    for i, cap in enumerate(caps):
        sel = np.isclose(c, cap)
        order = np.argsort(h[sel])
        ax.plot(hgrid, p.predict(cap, hgrid), color=RAMP[i], lw=1.5, alpha=0.9)
        ax.plot(h[sel][order], y[sel][order], "o", color=RAMP[i], ms=4.5, mec="white", mew=0.8)
        end = float(p.predict(cap, 1.0))
        if end - last_label >= 0.03:  # skip labels that would collide with the one below
            ax.annotate(NAMES[i], (1.0, end), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=MUTED)
            last_label = end
    ax.set_title(
        f"{method.value.capitalize()}: b1={p.capability_slope:.2f}, b2={p.hint_slope:.2f}, b0={p.intercept:.2f}, "
        f"A={p.upper:.3f}, L={p.lower:.3f}, F={p.floor:.4f}",
        fontsize=9,
        color=INK,
        loc="left",
    )
    ax.set_xlabel("Hint strength H")
    ax.set_xlim(-0.02, 1.12)
axes[0].set_ylabel("GSM8K accuracy (first 256 problems)")
fig.suptitle(
    "Delphi ladder: paper's observed accuracy (dots, mean over seeds) and our joint-law fit on all 12 rungs (lines)",
    x=0.01,
    ha="left",
    fontsize=11,
    color=INK,
)
fig.savefig(OUT / "delphi_hint_curves.png")
plt.close(fig)

# 2. Forecast of 1e23 from the first 9 rungs (up to 6e20).
t = 9
fit = fit_ladder(ladder.up_to(caps[t - 1]))
cgrid = np.linspace(caps[0] - 0.2, caps[-1] + 0.3, 200)
fig, ax = plt.subplots(figsize=(8, 4.6))
curves = {
    ForecastKind.MASKING: fit.joint[HintMethod.MASKING].predict(cgrid, 0.0),
    ForecastKind.TRUNCATION: fit.joint[HintMethod.TRUNCATION].predict(cgrid, 0.0),
    ForecastKind.SIGMOID_BASELINE: fit.sigmoid_baseline.predict(cgrid),
}
curves[ForecastKind.HINTED_ENSEMBLE] = (curves[ForecastKind.MASKING] + curves[ForecastKind.TRUNCATION]) / 2
for k in (ForecastKind.MASKING, ForecastKind.TRUNCATION, ForecastKind.HINTED_ENSEMBLE, ForecastKind.SIGMOID_BASELINE):
    ax.plot(cgrid, curves[k], color=SERIES[k], lw=2, ls="--" if k is ForecastKind.SIGMOID_BASELINE else "-", label=LABEL[k])
ax.plot(caps[:t], acc[:t], "o", color=INK, ms=6, mec="white", mew=1, label="Unhinted, fitted rungs", zorder=5)
ax.plot(caps[t:], acc[t:], "o", color="white", ms=7, mec=INK, mew=1.5, label="Unhinted, held out", zorder=5)
ax.axvline(caps[t - 1] + 0.05, color=MUTED, lw=0.8, ls=":")
ax.text(caps[t - 1] + 0.1, 0.5, "fit | extrapolate", color=MUTED, fontsize=8)
fc = fit.forecasts(caps[-1])
ax.annotate(
    f"1e23 observed {acc[-1]:.3f}\nmasking {fc[ForecastKind.MASKING]:.3f}, truncation {fc[ForecastKind.TRUNCATION]:.3f}\n"
    f"ensemble {fc[ForecastKind.HINTED_ENSEMBLE]:.3f}, sigmoid {fc[ForecastKind.SIGMOID_BASELINE]:.3f}",
    (caps[-1], acc[-1]),
    xytext=(22.15, 0.09),
    textcoords="data",
    fontsize=8,
    color=INK,
    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
)
ax.set_xticks(caps)
ax.set_xticklabels(NAMES, rotation=45, fontsize=8)
ax.set_xlabel("Training compute (FLOPs)")
ax.set_ylabel("Unhinted GSM8K accuracy")
ax.set_ylim(-0.01, 0.53)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("Forecasting the 1e23 model from the 9 smallest rungs (3e18 to 6e20)", loc="left", fontsize=11, color=INK)
fig.savefig(OUT / "delphi_forecast_t9.png")
plt.close(fig)

# 3. Forecast error vs number of fitted rungs, ours vs paper.
target, y_target = caps[-1], acc[-1]
ts = sorted(rp.PAPER_DELPHI_FORECAST)
ours = {ForecastKind.SIGMOID_BASELINE: [], ForecastKind.HINTED_ENSEMBLE: []}
for tt in ts:
    f = fit_ladder(ladder.up_to(caps[tt - 1])).forecasts(target)
    for k in ours:
        ours[k].append(abs(f[k] - y_target))
fig, ax = plt.subplots(figsize=(7, 4.2))
for k, idx in ((ForecastKind.SIGMOID_BASELINE, 0), (ForecastKind.HINTED_ENSEMBLE, 1)):
    ax.plot(ts, 100 * np.array(ours[k]), "-o", color=SERIES[k], lw=2, ms=7, mec="white", mew=1, label=f"{LABEL[k]} (ours)")
    ax.plot(ts, [100 * rp.PAPER_DELPHI_FORECAST[tt][idx] for tt in ts], "x", color=INK, ms=8, mew=1.5, label="Paper" if idx == 1 else None)
ax.set_xticks(ts)
ax.set_xticklabels([f"{tt}\n(to {NAMES[tt - 1]})" for tt in ts], fontsize=8)
ax.set_xlabel("Rungs fitted")
ax.set_ylabel("Absolute error forecasting 1e23 (pp)")
ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=8)
ax.set_title("1e23 forecast error, ours (lines) vs paper (x)", loc="left", fontsize=11, color=INK)
fig.savefig(OUT / "delphi_forecast_error.png")
plt.close(fig)
print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
