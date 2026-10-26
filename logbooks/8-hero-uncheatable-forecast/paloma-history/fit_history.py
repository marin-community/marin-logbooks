# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Historical Paloma floor diagnostics; CPU, no Hero loss used in fits."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

ROOT = Path(__file__).resolve().parent

records = json.loads((ROOT / "historical-inputs.json").read_text())


def matrices(rs):
    return (
        np.log(np.array([r["N_body"] for r in rs]) / 1e9),
        np.log(np.array([r["D"] for r in rs]) / 1e10),
        np.array([r["loss"] for r in rs]),
    )


def fit(rs, floor=None):
    x, z, y = matrices(rs)

    def pred(p):
        e, la, a, lb, b = p if floor is None else (floor, *p)
        return e + np.exp(la - a * x) + np.exp(lb - b * z)

    starts = []
    for e in [0.5, 1.5, 2.5]:
        if floor is None:
            p = [e, np.log(0.6), 0.3, np.log(0.6), 0.3]
            lo = [0, -10, 0.005, -10, 0.005]
            hi = [y.min() - 1e-5, 5, 2, 5, 2]
        else:
            p = [np.log(0.6), 0.3, np.log(0.6), 0.3]
            lo = [-10, 0.005, -10, 0.005]
            hi = [5, 2, 5, 2]
        sol = least_squares(lambda p: pred(p) - y, p, bounds=(lo, hi), max_nfev=2000, ftol=1e-12, gtol=1e-12, xtol=1e-12)
        starts.append(sol)
    sol = min(starts, key=lambda s: np.sum(s.fun**2))
    p = sol.x
    params = p.tolist() if floor is None else [floor, *p.tolist()]
    return dict(
        parameters=params,
        floor=params[0],
        rmse=float(np.sqrt(np.mean(sol.fun**2))),
        sse=float(np.sum(sol.fun**2)),
        count=len(rs),
        success=sol.success,
    )


def predict(m, rs):
    x, z, _ = matrices(rs)
    e, la, a, lb, b = m["parameters"]
    return e + np.exp(la - a * x) + np.exp(lb - b * z)


out = {}
variants = {
    "original81": [r for r in records if r["family"] == "delphi_original"],
    "all103": records,
    "no_d512": [r for r in records if r["width"] > 512],
    "repeat22": [r for r in records if r["family"] == "delphi_repeats"],
}
for label, rs in variants.items():
    m = fit(rs)
    fixed = fit(rs, 1.5)
    out[label] = dict(free=m, fixed15=fixed)
    print(label, out[label], flush=True)
    profiles = []
    for e in np.arange(0, 2.51, 0.1):
        profiles.append(fit(rs, float(e)))
    out[label]["profile"] = profiles
# Hold out the largest original budget (3e20) and all repeats of at least that compute.
base = variants["original81"]
train = [r for r in base if not r["id"].startswith("isoflop-3e+20-")]
test = [r for r in base if r["id"].startswith("isoflop-3e+20-")]
for floor in [None, 1.5]:
    m = fit(train, floor)
    p = predict(m, test)
    y = np.array([r["loss"] for r in test])
    out["heldout3e20_" + str(floor)] = dict(
        model=m, test_count=len(test), rmse=float(np.sqrt(np.mean((p - y) ** 2))), bias=float(np.mean(p - y))
    )
    print("HELDOUT", floor, out["heldout3e20_" + str(floor)], flush=True)
# Endpoint-only conventional C fit of isoFLOP minima is a different statistical question.
for label, rs in variants.items():
    grouped = {}
    for r in rs:
        # Same nominal budget+shape for originals; repeats are independently retrained minima.
        budget = r["id"].split("-")[1] if r["family"] == "delphi_original" else r["id"].split("-")[3]
        grouped.setdefault(budget, []).append(r)
    # use original seven scalar budget minima alone, without repeat labels or theoretical N,D terms.
    if label != "original81":
        continue
    mins = [min(group, key=lambda r: r["loss"]) for group in grouped.values()]
    x = np.log([r["compute_body"] for r in mins])
    y = np.array([r["loss"] for r in mins])
    center = x.mean()
    x -= center

    def objective(e, x=x, y=y):
        sl, ic = np.polyfit(x, np.log(y - e), 1)
        p = e + np.exp(ic + sl * x)
        return float(np.sum((p - y) ** 2))

    g = np.linspace(0, min(y) - 1e-5, 101)
    ix = np.argmin([objective(e) for e in g])
    sol = minimize_scalar(objective, bounds=(g[max(0, ix - 1)], g[min(100, ix + 1)]), method="bounded")
    floor = sol.x
    out["original_minima_C"] = dict(
        floor=float(floor), rmse=float(np.sqrt(objective(floor) / len(y))), ids=[r["id"] for r in mins]
    )
    print("MINIMA_C", out["original_minima_C"], flush=True)
(ROOT / "historical-fit-results.json").write_text(json.dumps(out, indent=2))
