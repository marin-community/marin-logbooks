# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit and repair Hero's pre-switch Uncheatable forecast from frozen public inputs.

Run with Python 3.12, numpy, scipy, and matplotlib. No network or accelerators.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
from scipy.optimize import minimize_scalar

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FRACTIONS = np.round(np.arange(0.30, 1.001, 0.05), 2)
STOP = 390251
CALIBRATION_END = 179999
UNCHEATABLE = "eval_dropless/uncheatable_eval/macro_loss"
PALOMA = "eval_dropless/paloma/macro_loss"


@dataclass(frozen=True)
class Group:
    mix: str
    fraction: float
    x: np.ndarray
    y: np.ndarray
    target_x: float


def group_fit(floor, group):
    residual = group.y - floor

    def objective(alpha):
        basis = np.exp(-alpha * group.x)
        amplitude = float(basis @ residual / (basis @ basis))
        return float(np.sum((floor + amplitude * basis - group.y) ** 2)), amplitude

    fit = minimize_scalar(lambda alpha: objective(alpha)[0], bounds=(0, 5), method="bounded")
    error, alpha, amplitude = min((objective(a)[0], a, objective(a)[1]) for a in (0, fit.x, 5))
    return error, float(floor + amplitude * np.exp(-alpha * group.target_x))


def curve(groups, floor):
    values = [group_fit(floor, g) for g in groups]
    rows = [{"mix": g.mix, "progress": g.fraction, "loss": p} for g, (_, p) in zip(groups, values, strict=True)]
    return rows, sum(error for error, _ in values)


def best_floor(groups, objective):
    upper = min(g.y.min() for g in groups) - 1e-7
    grid = np.linspace(0, upper, 31)
    scores = [objective(float(e)) for e in grid]
    index = int(np.argmin(scores))
    fit = minimize_scalar(objective, bounds=(grid[max(0, index - 1)], grid[min(30, index + 1)]), method="bounded")
    return min([(score, float(e)) for score, e in zip(scores, grid, strict=True)] + [(objective(fit.x), float(fit.x))])[
        1
    ]


def groups_for(data, metric, axis, max_width=1536, target_width=6144):
    runs = {(r["width"], r["mix"]): r for r in data["ladder"]}
    grids = {}
    for key, run in runs.items():
        grids[key] = {round(round((row["_step"] + 1) / run["stop"] * 20) / 20, 2): row for row in run["history"]}
    result = []
    target = runs[target_width, "old"]
    target_compute = target["compute"] * (1 - target["head_fraction"] if axis == "body" else 1)
    for mix in ("old", "new"):
        for f in FRACTIONS:
            widths = [w for w in (512, 768, 1024, 1536) if w <= max_width and f in grids[w, mix]]
            compute = np.array(
                [
                    runs[w, mix]["compute"] * f * (1 - runs[w, mix]["head_fraction"] if axis == "body" else 1)
                    for w in widths
                ]
            )
            center = np.log(compute).mean()
            y = np.array([grids[w, mix][f][metric] for w in widths])
            result.append(Group(mix, float(f), np.log(compute) - center, y, np.log(target_compute * f) - center))
    return result


def interpolate(rows, steps, mix="new"):
    selected = [r for r in rows if r["mix"] == mix]
    return np.interp((np.asarray(steps) + 1) / STOP, [r["progress"] for r in selected], [r["loss"] for r in selected])


def score(prediction, observed):
    error = np.asarray(prediction) - observed
    return {"rmse": float(np.sqrt(np.mean(error**2))), "bias": float(np.mean(error))}


def analyze(data, metric):
    steps = np.array([r["_step"] for r in data["hero"]])
    observed = np.array([r[metric] for r in data["hero"]])
    early = steps <= CALIBRATION_END
    models = {}
    for axis in ("total", "body"):
        groups = groups_for(data, metric, axis)
        floor = best_floor(groups, lambda e, gs=groups: curve(gs, e)[1])
        for mode, chosen_floor in [
            ("ladder", floor),
            (
                "early_hero",
                best_floor(
                    groups,
                    lambda e, gs=groups: float(
                        np.mean((interpolate(curve(gs, e)[0], steps[early]) - observed[early]) ** 2)
                    ),
                ),
            ),
        ]:
            values, fit_error = curve(groups, chosen_floor)
            p = interpolate(values, steps)
            models[mode + "_" + axis] = dict(
                floor=chosen_floor,
                ladder_sse=fit_error,
                curve=values,
                prediction=p.tolist(),
                test=score(p[~early], observed[~early]),
            )
        base = np.array(models["ladder_" + axis]["prediction"])
        offset = float(np.mean(observed[early] - base[early]))
        p = base + offset
        models["offset_" + axis] = dict(offset=offset, prediction=p.tolist(), test=score(p[~early], observed[~early]))
        # Holding out an entire width checks extrapolation without treating adjacent checkpoints as independent samples.
        small = groups_for(data, metric, axis, max_width=1024, target_width=1536)
        small_floor = best_floor(small, lambda e, gs=small: curve(gs, e)[1])
        small_rows, _ = curve(small, small_floor)
        actual = {
            (r["mix"], round(round((h["_step"] + 1) / r["stop"] * 20) / 20, 2)): h[metric]
            for r in data["ladder"]
            if r["width"] == 1536
            for h in r["history"]
        }
        paired = [
            (row["loss"], actual[row["mix"], row["progress"]])
            for row in small_rows
            if (row["mix"], row["progress"]) in actual
        ]
        models["ladder_" + axis]["d1536_holdout"] = dict(
            floor=small_floor, count=len(paired), **score(np.array(paired)[:, 0], np.array(paired)[:, 1])
        )
    original = [r for r in data["original_predictions"] if r["metric"] == metric and r["mixture"] == "new"]
    p = np.interp((steps + 1) / STOP, [r["progress"] for r in original], [r["loss"] for r in original])
    models["october3"] = dict(prediction=p.tolist(), test=score(p[~early], observed[~early]))
    return dict(
        metric=metric,
        steps=steps.tolist(),
        observed=observed.tolist(),
        calibration_count=int(early.sum()),
        test_count=int((~early).sum()),
        models=models,
    )


def render(results, subset_keys):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, key, title in zip(
        axes, (UNCHEATABLE, PALOMA), ("Uncheatable: 4K only", "Paloma control: 4K only"), strict=True
    ):
        r = results[key]
        ax.plot(np.array(r["steps"]) / 1000, r["observed"], color="#182a27", label="Measured Hero", linewidth=2)
        for name, label, color in [
            ("october3", "Oct 3 total-FLOP refit", "#b77a27"),
            ("ladder_body", "Body-FLOP refit; no Hero losses", "#226f9b"),
        ]:
            ax.plot(np.array(r["steps"]) / 1000, r["models"][name]["prediction"], label=label, color=color)
        ax.axvline(CALIBRATION_END / 1000, color="#888888", linestyle=":", linewidth=1)
        ax.set(title=title, xlabel="Training step (thousands)", ylabel="Mean loss (nats/token)")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "macro-comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    for ax, key in zip(axes.flat, subset_keys, strict=False):
        r = results[key]
        ax.plot(np.array(r["steps"]) / 1000, r["observed"], color="#182a27", label="Measured", linewidth=2)
        for name, label, color in [
            ("october3", "Oct 3 ladder fit", "#b77a27"),
            ("early_hero_total", "Floor calibrated through 180K", "#007e75"),
        ]:
            ax.plot(np.array(r["steps"]) / 1000, r["models"][name]["prediction"], color=color, label=label)
        ax.axvline(CALIBRATION_END / 1000, color="#888888", linestyle=":")
        ax.set_title(key.split("/")[-2].removesuffix("-llama3"))
        ax.set_xlabel("Training step (thousands)")
        ax.grid(alpha=0.2)
    axes.flat[-1].set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Subset forecasts: early calibration, later 4K holdout", y=0.995)
    fig.tight_layout()
    fig.savefig(ROOT / "subset-comparison.png", dpi=150)
    plt.close(fig)


def main():
    data = json.loads((ROOT / "inputs.json").read_text())
    keys = [k for k in data["metrics"] if k.endswith("/loss") and "/uncheatable_eval/" in k]
    assert len(keys) == 7
    results = {}
    for metric in [UNCHEATABLE, PALOMA, *keys]:
        results[metric] = analyze(data, metric)
        print(
            metric,
            {
                n: {k: v for k, v in m.items() if k in ("floor", "test", "d1536_holdout")}
                for n, m in results[metric]["models"].items()
            },
            flush=True,
        )
    observed = np.mean([results[k]["observed"] for k in keys], axis=0)
    assert np.max(np.abs(observed - results[UNCHEATABLE]["observed"])) < 3e-7
    derived = {}
    for name in (
        "october3",
        "ladder_total",
        "ladder_body",
        "early_hero_total",
        "early_hero_body",
        "offset_total",
        "offset_body",
    ):
        prediction = np.mean([results[k]["models"][name]["prediction"] for k in keys], axis=0)
        test = np.array(results[UNCHEATABLE]["steps"]) > CALIBRATION_END
        derived[name] = dict(prediction=prediction.tolist(), test=score(prediction[test], observed[test]))
    curves = []
    for f in FRACTIONS:
        row = {"progress": float(f), "step": round(float(f) * STOP)}
        for name in ("ladder_body", "early_hero_total", "early_hero_body"):
            row[name] = {
                k: next(
                    r["loss"] for r in results[k]["models"][name]["curve"] if r["mix"] == "new" and r["progress"] == f
                )
                for k in keys
            }
            row[name][UNCHEATABLE] = float(np.mean(list(row[name].values())))
        curves.append(row)
    output = dict(
        calibration_end=CALIBRATION_END,
        test_end=254999,
        target="seven-subset, 4K dropless loss; original schedule",
        results=results,
        derived_macro=derived,
        forecasts=curves,
    )
    (ROOT / "results.json").write_text(json.dumps(output, indent=2))
    print("DERIVED MACRO", {k: v["test"] for k, v in derived.items()})
    render(results, keys)


if __name__ == "__main__":
    main()
