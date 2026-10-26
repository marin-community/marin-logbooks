# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent
records = json.loads((ROOT / "67b-inputs.json").read_text())
small = records[:3]
main = records[3]

out = {}
for low, high in [(0.3, 0.8), (0.3, 0.65), (0.5, 0.8)]:
    groups = []
    for f in np.round(np.arange(low, high + 0.001, 0.05), 2):
        C = []
        loss = []
        for r in small:
            h = r["history"]
            fs = [(row["_step"] + 1) / r["stop"] for row in h]
            ys = [row["eval/paloma/macro_loss"] for row in h]
            C.append(r["body_fpt"] * r["total_tokens"] * f)
            loss.append(np.interp(f, fs, ys))
        groups.append((np.log(C), np.array(loss), f))
    upper = min(y.min() for x, y, f in groups) - 1e-5

    def curves(e, groups=groups):
        err = 0
        rows = []
        for x, y, f in groups:
            center = x.mean()
            x = x - center
            sl, ic = np.polyfit(x, np.log(y - e), 1)
            p = e + np.exp(ic + sl * x)
            err += float(np.sum((p - y) ** 2))
            pred = float(e + np.exp(ic + sl * (np.log(main["body_fpt"] * main["total_tokens"] * f) - center)))
            rows.append(dict(f=f, pred=pred))
        return err, rows

    grid = np.linspace(0, upper, 101)
    i = np.argmin([curves(e)[0] for e in grid])
    sol = minimize_scalar(lambda e: curves(e)[0], bounds=(grid[max(0, i - 1)], grid[min(100, i + 1)]), method="bounded")
    e = sol.x
    for label, floor in [("free", e), ("fixed14", 1.4), ("fixed15", 1.5)]:
        err, rows = curves(floor)
        steps = np.array([r["_step"] for r in main["history"]])
        D = (15000 * 4096 + (steps + 1 - 15000) * 8192) * 8192
        f = D / main["total_tokens"]
        mask = (f >= low) & (f <= high) & (steps < 127500)
        p = np.interp(f[mask], [r["f"] for r in rows], [r["pred"] for r in rows])
        y = np.array([r["eval/paloma/macro_loss"] for r in main["history"]])[mask]
        out[f"{low}_{high}_{label}"] = dict(
            floor=float(floor),
            ladder_rmse=float(np.sqrt(err / (len(groups) * 3))),
            main_n=int(mask.sum()),
            main_rmse=float(np.sqrt(np.mean((p - y) ** 2))),
            main_bias=float(np.mean(p - y)),
            curve=rows,
        )
        print(low, high, label, {k: v for k, v in out[f"{low}_{high}_{label}"].items() if k != "curve"})
(ROOT / "67b-fit-results.json").write_text(json.dumps(out, indent=2))
