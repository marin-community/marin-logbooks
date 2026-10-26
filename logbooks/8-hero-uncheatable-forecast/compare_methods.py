# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare original Uncheatable, Paloma-style, and fitted-floor forecasts."""

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
from forecast import PALOMA, STOP, UNCHEATABLE, best_floor, groups_for, interpolate, score

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def log_curve(groups, floor):
    rows = []
    sse = 0.0
    for group in groups:
        assert np.all(group.y > floor)
        slope, intercept = np.polyfit(group.x, np.log(group.y - floor), 1)
        fitted = floor + np.exp(intercept + slope * group.x)
        sse += float(np.sum((fitted - group.y) ** 2))
        rows.append(
            dict(
                mix=group.mix,
                progress=group.fraction,
                loss=float(floor + np.exp(intercept + slope * group.target_x)),
                alpha=float(-slope),
            )
        )
    return rows, sse


def compare(data):
    output = dict(hero_loss_used=False, target="4K, seven-subset Uncheatable macro loss, nats/token", metrics={})
    steps = np.array([r["_step"] for r in data["hero"]])
    for metric in (UNCHEATABLE, PALOMA):
        observed = np.array([r[metric] for r in data["hero"]])
        result = dict(steps=steps.tolist(), observed=observed.tolist(), models={})
        for axis in ("total", "body"):
            groups = groups_for(data, metric, axis)
            fixed_floor = 1.4 if metric == UNCHEATABLE else 1.5
            inferred = best_floor(groups, lambda e, gs=groups: log_curve(gs, e)[1])
            for method, floor in (("fixed", fixed_floor), ("fit", inferred)):
                rows, sse = log_curve(groups, floor)
                p = interpolate(rows, steps)
                result["models"][f"{axis}_{method}"] = dict(
                    floor=floor,
                    ladder_sse=sse,
                    curve=rows,
                    prediction=p.tolist(),
                    all_4k=score(p, observed),
                    later_4k=score(p[steps > 179999], observed[steps > 179999]),
                )
        # The published lines are the actual baselines, not reconstructed approximations.
        legacy_id = (
            "hero-uncheatable-floor14-new-from30-75c36f" if metric == UNCHEATABLE else "hero-forecast-seqlen4096-fc"
        )
        legacy = [r for r in data["legacy"][legacy_id] if metric in r]
        p = np.interp(steps, [r["_step"] for r in legacy], [r[metric] for r in legacy])
        result["models"]["published"] = dict(
            source=legacy_id,
            prediction=p.tolist(),
            all_4k=score(p, observed),
            later_4k=score(p[steps > 179999], observed[steps > 179999]),
        )
        # Also retain the October 3 revision, which changed loss-space fitting as well as the floor.
        refit = [r for r in data["original_predictions"] if r["metric"] == metric and r["mixture"] == "new"]
        p = np.interp((steps + 1) / STOP, [r["progress"] for r in refit], [r["loss"] for r in refit])
        result["models"]["october3"] = dict(
            prediction=p.tolist(), all_4k=score(p, observed), later_4k=score(p[steps > 179999], observed[steps > 179999])
        )
        output["metrics"][metric] = result
    return output


def render(output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in zip(
        axes, (UNCHEATABLE, PALOMA), ("Uncheatable: pre-switch 4K", "Paloma control: pre-switch 4K"), strict=True
    ):
        r = output["metrics"][metric]
        x = np.array(r["steps"]) / 1000
        ax.plot(x, r["observed"], color="#18342f", linewidth=2.4, label="Measured Hero")
        for method, label, color in (
            ("published", "Published forecast", "#b57a29"),
            ("body_fixed", "Paloma method; existing corpus floor", "#be5454"),
            ("body_fit", "Paloma method; fitted corpus floor", "#206d99"),
        ):
            ax.plot(x, r["models"][method]["prediction"], color=color, label=label, linewidth=1.6)
        ax.set(title=title, xlabel="Training step (thousands)", ylabel="Mean loss (nats/token)")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Same scoring window; no Hero losses used to fit any forecast", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "method-comparison.png", dpi=180)
    plt.close(fig)


def main():
    data = json.loads((ROOT / "inputs.json").read_text())
    output = compare(data)
    (ROOT / "method-results.json").write_text(json.dumps(output, indent=2))
    render(output)
    for metric, result in output["metrics"].items():
        print(metric)
        for name, model in result["models"].items():
            print(name, {k: v for k, v in model.items() if k in ("floor", "ladder_sse", "all_4k", "later_4k")})


if __name__ == "__main__":
    main()
