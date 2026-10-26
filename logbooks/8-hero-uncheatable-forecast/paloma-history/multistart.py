# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
rs = [r for r in json.loads((ROOT / "historical-inputs.json").read_text()) if r["family"] == "delphi_original"]
x = np.log(np.array([r["N_body"] for r in rs]) / 1e9)
z = np.log(np.array([r["D"] for r in rs]) / 1e10)
y = np.array([r["loss"] for r in rs])
sols = []


def prediction(p):
    e, la, a, lb, b = p
    return e + np.exp(la - a * x) + np.exp(lb - b * z)


for e in [0.1, 1, 1.5, 2, 2.5]:
    for a, b in [(0.05, 0.05), (0.1, 0.4), (0.4, 0.1), (0.8, 0.8), (1.5, 0.1), (0.1, 1.5)]:
        q = least_squares(
            lambda p: prediction(p) - y,
            [e, np.log(0.6), a, np.log(0.6), b],
            bounds=([0, -10, 0.005, -10, 0.005], [y.min() - 1e-5, 5, 2, 5, 2]),
            max_nfev=3000,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        sols.append(q)
best = min(sols, key=lambda q: np.sum(q.fun**2))
r = dict(
    start_count=len(sols),
    best_parameters=best.x.tolist(),
    best_sse=float(np.sum(best.fun**2)),
    near_minimum_solutions=int(sum(np.sum(s.fun**2) < np.sum(best.fun**2) * 1.00001 for s in sols)),
    floors_near_minimum=[float(s.x[0]) for s in sols if np.sum(s.fun**2) < np.sum(best.fun**2) * 1.00001],
    jacobian_singular_values=np.linalg.svd(best.jac, compute_uv=False).tolist(),
    bounds=dict(lower=[0, -10, 0.005, -10, 0.005], upper=[float(y.min() - 1e-5), 5, 2, 5, 2]),
)
(ROOT / "multistart-results.json").write_text(json.dumps(r, indent=2))
print(json.dumps(r))
