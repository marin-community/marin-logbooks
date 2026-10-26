# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared floor, separate family/progress slopes; no Hero data fitted."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent
raw = json.loads((ROOT / "rav-inputs.json").read_text())["rungs"]
keys = list(raw)
compute = dict(
    d768=2.649679880270119e19, d1024=1.7261182741798453e20, d1536=1.3813258500033133e21, d2048=7.741814947656449e21
)
tpu = []
for f in np.round(np.arange(0.3, 0.801, 0.05), 2):
    x = []
    y = []
    for k, r in raw.items():
        grid = {
            round(round(h["_step"] / r["stop"] * 20) / 20, 2): h["eval_dropless/paloma/macro_loss"] for h in r["history"]
        }
        x.append(compute[k] * f)
        y.append(grid[f])
    tpu.append((np.log(x) - np.log(x).mean(), np.array(y)))
rs = json.loads((ROOT / "historical-inputs.json").read_text())
models = json.loads((ROOT / "models.json").read_text())
selected = [r for r in rs if r["id"] in [m["source_id"] for m in models.values()]]
repeat = [r for r in rs if r["family"] == "delphi_repeats"]
groups = {}
for r in selected:
    slug = next(k for k, v in models.items() if v["source_id"] == r["id"])
    groups.setdefault(slug, []).append(r)
for r in repeat:
    slug = r["id"].split("-")[3]
    seed = int(r["id"].split("-")[4][4:])
    if seed == 0 and slug in ["3e18", "9e18"]:
        continue
    groups.setdefault(slug, []).append(r)
means = [
    dict(
        slug=k,
        loss=float(np.mean([r["loss"] for r in group])),
        compute_body=float(np.mean([r["compute_body"] for r in group])),
        count=len(group),
        ids=[r["id"] for r in group],
    )
    for k, group in groups.items()
]
x = np.log([r["compute_body"] for r in means])
delphi = [(x - x.mean(), np.array([r["loss"] for r in means]))]
base = [r for r in rs if r["family"] == "delphi_original"]
budgets = {}
for r in base:
    budgets.setdefault(r["id"].split("-")[1], []).append(r)
mins = [min(g, key=lambda r: r["loss"]) for g in budgets.values()]
x = np.log([r["compute_body"] for r in mins])
minimum = [(x - x.mean(), np.array([r["loss"] for r in mins]))]


def err(gs, e):
    ss = 0
    n = 0
    for x, y in gs:
        slope, intercept = np.polyfit(x, np.log(y - e), 1)
        p = e + np.exp(intercept + slope * x)
        ss += float(np.sum((p - y) ** 2))
        n += len(y)
    return ss, n


out = dict(
    group_counts=dict(
        TPU_progress_groups=len(tpu),
        TPU_points=sum(len(y) for x, y in tpu),
        Delphi_endpoint_budgets=len(means),
        Delphi_runs=sum(r["count"] for r in means),
    ),
    delphi_budget_means=means,
    fits={},
)
for variant, dlp in [("budget_means27", delphi), ("best_at_budget7", minimum)]:
    for weight in [0, 0.25, 0.5, 0.75, 1.0]:

        def objective(e, dlp=dlp, weight=weight):
            a, n = err(tpu, e)
            b, m = err(dlp, e)
            return (1 - weight) * a / n + weight * b / m

        upper = min(y.min() for x, y in tpu + dlp) - 1e-5
        g = np.linspace(0, upper, 201)
        i = np.argmin([objective(e) for e in g])
        opt = minimize_scalar(objective, bounds=(g[max(0, i - 1)], g[min(200, i + 1)]), method="bounded")
        floor = float(opt.x)
        a, n = err(tpu, floor)
        b, m = err(dlp, floor)
        out["fits"][variant + "_" + str(weight)] = dict(
            floor=floor,
            Delphi_weight=weight,
            objective=objective(floor),
            tpu_rmse=float(np.sqrt(a / n)),
            delphi_rmse=float(np.sqrt(b / m)),
        )
        print(variant, weight, out["fits"][variant + "_" + str(weight)])
        if weight == 0.5:
            out["fits"][variant + "_" + str(weight)]["profile"] = [
                dict(floor=float(e), objective=objective(e)) for e in g
            ]

    # Rawperpoint pooled fit.
    def objective(e, dlp=dlp):
        return err(tpu, e)[0] + err(dlp, e)[0]

    opt = minimize_scalar(objective, bounds=(0, upper), method="bounded")
    out["fits"][variant + "_raw_points"] = dict(floor=float(opt.x), sse=objective(opt.x))
    print("RAW", variant, out["fits"][variant + "_raw_points"])
(ROOT / "shared-floor-results.json").write_text(json.dumps(out, indent=2))
