# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Profile the original Paloma ladder's floor, without fitting any Hero data."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
from scipy.optimize import brentq, minimize_scalar

mpl.use("Agg")
import matplotlib.pyplot as plt

METRIC = "eval_dropless/paloma/macro_loss"
RUNS = {
    "d768": "rav-ladder-d768-v2",
    "d1024": "rav-ladder-d1024",
    "d1536": "rav-ladder-d1536",
    "d2048": "rav-ladder-d2048-v3",
}
# Exact lm_head-excluded compute constants from the original d23e6e9 script.
COMPUTE = {
    "d768": 2.649679880270119e19,
    "d1024": 1.7261182741798453e20,
    "d1536": 1.3813258500033133e21,
    "d2048": 7.741814947656449e21,
    "d6144": 2.6108941613772703e24,
}
STOP = 390251
FRACTIONS = np.round(np.arange(0.05, 1.0001, 0.05), 2)


def read_data():
    raw = json.loads(Path(__file__).with_name("rav-inputs.json").read_text())["rungs"]
    return {
        name: dict(**r, grid={round(round(row["_step"] / r["stop"] * 20) / 20, 2): row[METRIC] for row in r["history"]})
        for name, r in raw.items()
    }


def fit_at(data, floor, fraction, keep=tuple(RUNS)):
    window = [0.6, 0.65, 0.7, 0.75, 0.8]
    slope, intercept = np.polyfit(window, [data["d2048"]["grid"][f] for f in window], 1)
    final_measured = max(data["d2048"]["grid"])
    losses = []
    for rung in keep:
        if rung == "d2048" and fraction > final_measured:
            losses.append(float(slope * fraction + intercept + 0.005))
        else:
            losses.append(data[rung]["grid"][fraction])
    losses = np.array(losses)
    assert np.all(losses > floor)
    compute = np.array([COMPUTE[r] * fraction for r in keep])
    log_compute = np.log(compute)
    center = log_compute.mean()
    slope, intercept = np.polyfit(log_compute - center, np.log(losses - floor), 1)
    fitted = floor + np.exp(intercept + slope * (log_compute - center))
    predicted = floor + np.exp(intercept + slope * (np.log(COMPUTE["d6144"] * fraction) - center))
    return dict(
        progress=float(fraction),
        loss=float(predicted),
        alpha=float(-slope),
        ladder_sse=float(np.sum((fitted - losses) ** 2)),
        fitted=fitted.tolist(),
        observed=losses.tolist(),
    )


def fit_floor(data, fractions, keep=tuple(RUNS)):
    upper = min(min(fit_at(data, 0, f, keep)["observed"]) for f in fractions) - 1e-6

    def objective(floor):
        return sum(fit_at(data, floor, f, keep)["ladder_sse"] for f in fractions)

    grid = np.linspace(0, upper, 201)
    scores = [objective(float(e)) for e in grid]
    i = int(np.argmin(scores))
    fit = minimize_scalar(
        objective, bounds=(grid[max(0, i - 1)], grid[min(200, i + 1)]), method="bounded", options={"xatol": 1e-12}
    )
    sse, floor = min(
        [(score, float(e)) for score, e in zip(scores, grid, strict=True)] + [(objective(fit.x), float(fit.x))]
    )
    return floor, sse, [dict(floor=float(e), sse=float(s)) for e, s in zip(grid, scores, strict=True)]


def score(curve, hero):
    steps = np.array([r["_step"] for r in hero])
    actual = np.array([r[METRIC] for r in hero])
    forecast = np.interp((steps + 1) / STOP, [r["progress"] for r in curve], [r["loss"] for r in curve])
    output = {}
    for name, mask in [("all_4k", np.ones(len(steps), dtype=bool)), ("later_4k", steps > 179999)]:
        err = forecast[mask] - actual[mask]
        output[name] = dict(
            count=int(mask.sum()),
            rmse=float(np.sqrt(np.mean(err**2))),
            bias=float(err.mean()),
            mae=float(np.abs(err).mean()),
        )
    return dict(
        **output, prediction=forecast.tolist(), last_prediction=float(forecast[-1]), last_observed=float(actual[-1])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=Path(__file__).resolve().parent.parent / "inputs.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results.json")
    args = parser.parse_args()
    data = read_data()
    frozen = json.loads(args.inputs.read_text())
    hero = frozen["hero"]
    result = dict(
        hero_loss_used=False,
        source="original Paloma rav-ladder",
        metric=METRIC,
        units="nats/token",
        compute_excludes_lm_head=True,
        source_counts={
            r: dict(evals=len(d["grid"]), max_progress=max(d["grid"]), stop=d["stop"]) for r, d in data.items()
        },
        models={},
    )
    windows = {
        "fit_30_80": FRACTIONS[(FRACTIONS >= 0.3) & (FRACTIONS <= 0.8)],
        "fit_05_80": FRACTIONS[FRACTIONS <= 0.8],
        "fit_30_100_including_extrapolation": FRACTIONS[FRACTIONS >= 0.3],
        "fit_05_100_including_extrapolation": FRACTIONS,
    }
    fixed = [fit_at(data, 1.5, f) for f in FRACTIONS]
    result["models"]["fixed_1.5"] = dict(
        floor=1.5,
        curve=fixed,
        final=fixed[-1]["loss"],
        ladder_sse=sum(r["ladder_sse"] for r in fixed),
        **score(fixed, hero),
    )
    for name, fractions in windows.items():
        floor, sse, profile = fit_floor(data, fractions)
        curve = [fit_at(data, floor, f) for f in FRACTIONS]
        result["models"][name] = dict(
            floor=floor,
            fit_fractions=fractions.tolist(),
            ladder_sse=sse,
            fixed_1_5_sse_same_window=sum(fit_at(data, 1.5, f)["ladder_sse"] for f in fractions),
            curve=curve,
            final=curve[-1]["loss"],
            profile=profile,
            **score(curve, hero),
        )
    # A finite-floor sensitivity, not a confidence interval.
    for floor in [1.3, 1.4, 1.5, 1.6, 1.7]:
        if floor >= min(data[r]["grid"][f] for r in RUNS for f in FRACTIONS if f in data[r]["grid"]):
            continue
        curve = [fit_at(data, floor, f) for f in FRACTIONS]
        result["models"]["sensitivity_" + str(floor)] = dict(floor=floor, final=curve[-1]["loss"], **score(curve, hero))
    optimum = result["models"]["fit_30_80"]
    fit_fractions = optimum["fit_fractions"]

    def objective(e):
        return sum(fit_at(data, e, t)["ladder_sse"] for t in fit_fractions)

    threshold = optimum["ladder_sse"] * 1.1
    floor = optimum["floor"]
    upper_bound = min(min(fit_at(data, 0, t)["observed"]) for t in fit_fractions) - 1e-6
    low = brentq(lambda e: objective(e) - threshold, 0, floor)
    high = brentq(lambda e: objective(e) - threshold, floor, upper_bound)
    result["measured_30_80_profile_summary"] = dict(
        fitted_floor=floor,
        opt_sse=optimum["ladder_sse"],
        fixed_1_5_sse=objective(1.5),
        fixed_to_opt_sse_ratio=objective(1.5) / optimum["ladder_sse"],
        within_10pct_additional_sse_floor_range=[low, high],
        qualification="Floor-sensitivity range only, not a confidence interval.",
    )
    # Independently recover the source prediction with centered one-variable normal equations.
    for fraction in [0.3, 0.5, 0.65, 0.8, 1.0]:
        row = fit_at(data, 1.5, fraction)
        x = np.log(np.array([COMPUTE[rung] * fraction for rung in RUNS]))
        y = np.log(np.array(row["observed"]) - 1.5)
        slope = ((x - x.mean()) @ (y - y.mean())) / ((x - x.mean()) @ (x - x.mean()))
        direct = 1.5 + np.exp(y.mean() + slope * (np.log(COMPUTE["d6144"] * fraction) - x.mean()))
        assert abs(direct - row["loss"]) < 1e-12
    assert abs(result["models"]["fixed_1.5"]["final"] - 2.039) < 0.0005
    assert result["source_counts"]["d2048"]["max_progress"] == 0.8
    assert len(hero) == 46
    result["checks"] = {
        "normal_equation_parity": True,
        "published_endpoint_replication": True,
        "only_measured_30_80_for_main_floor": True,
        "hero_scoring_count": 46,
    }
    args.output.write_text(json.dumps(result, indent=2))
    render(result, hero, args.output.with_suffix(".png"))
    print(
        json.dumps(
            {
                n: {
                    k: v
                    for k, v in m.items()
                    if k in ("floor", "final", "all_4k", "later_4k", "ladder_sse", "last_prediction")
                }
                for n, m in result["models"].items()
            },
            indent=2,
        )
    )


def render(result, hero, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    steps = np.array([r["_step"] for r in hero]) / 1000
    axes[0].plot(steps, [r[METRIC] for r in hero], label="Measured Hero at 4K", color="#18342f", linewidth=2.2)
    for name, label, color in [
        ("fixed_1.5", "Original fixed floor 1.5", "#b57a29"),
        ("fit_30_80", "Ladder-fitted floor 1.291", "#206d99"),
    ]:
        model = result["models"][name]
        axes[0].plot(steps, model["prediction"], label=label, color=color)
    axes[0].set(
        title="Original Paloma ladder: same 4K Hero window",
        xlabel="Training step (thousands)",
        ylabel="Paloma macro loss (nats/token)",
    )
    axes[0].legend(fontsize=8)
    profile = result["models"]["fit_30_80"]["profile"]
    p = [r for r in profile if 0.9 <= r["floor"] <= 1.7]
    minimum = result["models"]["fit_30_80"]["ladder_sse"]
    axes[1].plot([r["floor"] for r in p], [r["sse"] / minimum for r in p], color="#206d99")
    axes[1].axvline(1.5, color="#b57a29", linestyle="--", label="Original floor 1.5")
    low, high = result["measured_30_80_profile_summary"]["within_10pct_additional_sse_floor_range"]
    axes[1].axvspan(low, high, color="#206d99", alpha=0.15, label="≤10% extra fit SSE; not a CI")
    axes[1].set(
        title="Floor profile from measured 30-80% ladder points",
        xlabel="Assumed floor (nats/token)",
        ylabel="Fit SSE / best fit SSE",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
