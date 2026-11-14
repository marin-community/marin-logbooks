# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze shared matrix-envelope coefficients using calibration responses only."""

import argparse
import hashlib
import json
from pathlib import Path

import cvxpy as cp
import numpy as np

from experiments.ema_gram.geometry_math import normalized_hessian_norm


def components(row, control):
    base = np.asarray(row["envelope_components"], np.float64)
    if control == "scalar":
        return np.stack([base[0], row["gradient_norm"] ** 2 * base[0]])
    if control == "oriented":
        return base
    if control == "rotated":
        return np.asarray(row["rotated_components"], np.float64)
    raise ValueError(control)


def fit_envelope(rows, control):
    """Minimize mean subspace trace with both signed PSD constraints at each state."""
    hessians = [np.asarray(r["projected_hessian"], np.float64) for r in rows]
    hessians = [(h + h.T) / 2 for h in hessians]
    bases = [components(r, control) for r in rows]
    hscale = max(max(np.linalg.norm(h, 2) for h in hessians), 1e-30)
    scales = np.mean([[np.trace(c) / len(c) for c in b] for b in bases], axis=0)
    # A zero gradient component carries no information; scaling it by one is benign.
    scales = np.where(scales > 1e-30, scales, 1.0)
    variable = cp.Variable(len(scales), nonneg=True)
    constraints = [variable[0] >= 1e-6 * scales[0]]
    objectives = []
    for h, b in zip(hessians, bases, strict=True):
        envelope = sum(variable[i] * b[i] / scales[i] for i in range(len(b)))
        constraints.extend([envelope - h / hscale >> 0, envelope + h / hscale >> 0])
        objectives.append(cp.trace(envelope) / len(h))
    problem = cp.Problem(cp.Minimize(sum(objectives) / len(objectives)), constraints)
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_feas=1e-9, tol_gap_rel=1e-9)
    if problem.status != cp.OPTIMAL or variable.value is None:
        raise ValueError(f"Envelope solver did not converge: {problem.status}")
    coefficients = np.maximum(variable.value, 0) * hscale / scales
    coefficients[0] = max(coefficients[0], 1e-6 * hscale)
    padding = 0.0
    for h, b in zip(hessians, bases, strict=True):
        envelope = np.einsum("i,ijk->jk", coefficients, b)
        minimum = min(np.linalg.eigvalsh(envelope + h).min(), np.linalg.eigvalsh(envelope - h).min())
        padding = max(padding, (1e-9 * hscale - minimum) / np.linalg.eigvalsh(b[0]).min())
    if padding > 1e-5 * hscale:
        raise ValueError("Solver feasibility error is too large to round conservatively")
    coefficients[0] += padding
    checks = []
    for h, b in zip(hessians, bases, strict=True):
        envelope = np.einsum("i,ijk->jk", coefficients, b)
        checks.append(
            dict(
                mean_trace=float(np.trace(envelope) / len(h)),
                normalized_hessian_norm=normalized_hessian_norm(h, envelope),
            )
        )
    if max(c["normalized_hessian_norm"] for c in checks) > 1 + 1e-7:
        raise ValueError("Frozen envelope does not cover its calibration subspaces")
    return dict(
        control=control,
        coefficients=coefficients.tolist(),
        coefficient_names=["L0", "c"] if control == "scalar" else ["L0", "cL", "cR"],
        intercept_floor=1e-6 * hscale,
        feasibility_intercept_padding=padding,
        solver="CLARABEL",
        solver_status=problem.status,
        cvxpy_version=cp.__version__,
        calibration_checks=checks,
        calibration_mean_trace=float(np.mean([c["mean_trace"] for c in checks])),
    )


def read_screen(directory):
    def read(name):
        return json.loads((directory / name).read_text())

    complete, result = read("completed.json"), read("envelope-results.json")
    if not complete["success"] or complete["stage"] != "screen" or complete["experiment"] != "envelope":
        raise ValueError("Only completed calibration screens may enter a fit")
    if result["stage"] != "screen" or result["state"] != complete["state"] or len(result["rows"]) != 2:
        raise ValueError("Incomplete or mismatched screen")
    if result["data_manifest_sha256"] != complete["data_manifest_sha256"]:
        raise ValueError("Screen manifest identity mismatch")
    for row in result["rows"]:
        if row["symmetry_relative"] >= 1e-4 or not all(c["passed"] for c in row["derivative_checks"]):
            raise ValueError("Unvalidated Hessian screen")
        if np.asarray(row["projected_hessian"]).shape != (8, 8):
            raise ValueError("Expected full eight-dimensional projected Hessian")
    return complete, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screens", nargs=2, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    screens = [read_screen(d) for d in args.screens]
    if {c["state"] for c, _ in screens} != {"eighth-middle", "quarter-middle"}:
        raise ValueError("Require both prespecified middle checkpoints exactly once")
    if len({c["data_manifest_sha256"] for c, _ in screens}) != 1:
        raise ValueError("Shared fresh reference panel changed across states")
    plan = dict(
        scope="Calibration only; freeze and publish before test responses", inputs={}, fits={}, sources=[], path_gate={}
    )
    for directory, (complete, result) in zip(args.screens, screens, strict=True):
        plan["inputs"][complete["state"]] = {r["block"]: r["identities"] for r in result["rows"]}
        plan["sources"].append(
            dict(
                completed=complete,
                result_sha256=hashlib.sha256((directory / "envelope-results.json").read_bytes()).hexdigest(),
            )
        )
    for block in ("attention", "mlp"):
        rows = [next(r for r in result["rows"] if r["block"] == block) for _, result in screens]
        fits = {control: fit_envelope(rows, control) for control in ("scalar", "oriented", "rotated")}
        plan["fits"][block] = fits
        ratio = fits["oriented"]["calibration_mean_trace"] / fits["scalar"]["calibration_mean_trace"]
        plan["path_gate"][block] = dict(
            oriented_to_scalar_trace=ratio,
            eligible=ratio <= 0.95,
            rule="At least 5% calibration trace reduction; no test outcomes used",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Do not overwrite a frozen plan; choose a new explicit version")
    args.output.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    print(json.dumps(plan["path_gate"], indent=2))


if __name__ == "__main__":
    main()
