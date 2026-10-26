# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent
rs = json.loads((ROOT / "historical-inputs.json").read_text())
models = json.loads((ROOT / "models.json").read_text())
trajectories = {r["id"]: r["history"] for r in json.loads((ROOT / "historical-trajectories.json").read_text())}


def curve_floor(x, y):
    x = np.log(x)
    center = x.mean()
    x -= center
    y = np.array(y)

    def calc(e, groups=groups):
        slope, intercept = np.polyfit(x, np.log(y - e), 1)
        p = e + np.exp(intercept + slope * x)
        return float(np.sum((p - y) ** 2))

    grid = np.linspace(0, min(y) - 1e-5, 151)
    i = np.argmin([calc(e) for e in grid])
    opt = minimize_scalar(calc, bounds=(grid[max(0, i - 1)], grid[min(150, i + 1)]), method="bounded")
    e = opt.x
    return dict(
        floor=float(e),
        rmse=float(np.sqrt(calc(e) / len(y))),
        fixed15_rmse=float(np.sqrt(calc(1.5) / len(y))),
        count=len(y),
        floor_profile=[
            dict(floor=float(v), rmse=float(np.sqrt(calc(v) / len(y)))) for v in np.arange(0, min(2.51, min(y)), 0.1)
        ],
    )


original = [r for r in rs if r["family"] == "delphi_original"]
selected = [r for r in original if r["id"] in [m["source_id"] for m in models.values()]]
reps = [r for r in rs if r["family"] == "delphi_repeats"]
gr = {}
for r in reps:
    gr.setdefault(r["id"].split("-")[3], []).append(r)
means = [
    dict(
        compute_body=np.mean([r["compute_body"] for r in a]),
        compute_total=np.mean([r["compute_total"] for r in a]),
        loss=np.mean([r["loss"] for r in a]),
    )
    for a in gr.values()
]
out = {}
for name, r in [("selected7", selected), ("repeat_means9", means)]:
    for axis in ["compute_body", "compute_total"]:
        out[name + "_" + axis] = curve_floor([x[axis] for x in r], [x["loss"] for x in r])
        print(name, axis, {k: v for k, v in out[name + "_" + axis].items() if k != "floor_profile"})
# Sharedfloor across actual checkpoints of Delphi selected or repeats, interpolated onto each run's progress.
for label, rids in [("selected7", [r["id"] for r in selected]), ("repeat22", [r["id"] for r in reps])]:
    for axis in ["compute_body", "compute_total"]:
        for low, high in [(0.3, 0.8), (0.3, 1.0)]:
            groups = []
            for f in np.round(np.arange(low, high + 0.001, 0.05), 2):
                xs = []
                ys = []
                for rid in rids:
                    r = next(r for r in rs if r["id"] == rid)
                    h = trajectories[rid]
                    fr = np.array([(row["_step"] + 1) / r["stop"] for row in h])
                    yv = np.array([row["eval/paloma/macro_loss"] for row in h])
                    if fr.min() <= f <= fr.max():
                        xs.append(r[axis] * f)
                        ys.append(float(np.interp(f, fr, yv)))
                groups.append((np.log(xs) - np.log(xs).mean(), np.array(ys)))
            upper = min(y.min() for x, y in groups) - 1e-5

            def calc(e, groups=groups):
                err = 0
                for x, y in groups:
                    sl, ic = np.polyfit(x, np.log(y - e), 1)
                    err += float(np.sum((e + np.exp(ic + sl * x) - y) ** 2))
                return err

            grid = np.linspace(0, upper, 101)
            i = np.argmin([calc(e) for e in grid])
            opt = minimize_scalar(calc, bounds=(grid[max(0, i - 1)], grid[min(100, i + 1)]), method="bounded")
            n = sum(len(y) for x, y in groups)
            out[f"{label}_{axis}_{low}_{high}"] = dict(
                floor=float(opt.x), rmse=float(np.sqrt(calc(opt.x) / n)), count=n, progress=[low, high]
            )
            print(label, axis, low, high, out[f"{label}_{axis}_{low}_{high}"])
(ROOT / "historical-curve-results.json").write_text(json.dumps(out, indent=2))
