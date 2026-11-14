# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate fixed-coefficient path evidence and summarize sampled coverage."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.ema_gram.geometry_math import normalized_hessian_norm


def analyze_paths(directories, plan, inputs):
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    if inputs["plan_sha256"] != digest:
        raise ValueError("Path inputs use a different coefficient plan")
    measurements, losses, completions = [], [], []
    states = set()
    expected_points = {(None, 0.0)} | {(a, t) for a in (0.125, 0.25) for t in (0.5, 1.0)}
    expected_controls = {(r, c) for r in ("starting", "current") for c in ("scalar", "oriented", "rotated")}
    for directory in directories:
        complete = json.loads((directory / "completed.json").read_text())
        result = json.loads((directory / "envelope-path-results.json").read_text())
        frozen = json.loads((directory / "frozen-envelope-plan.json").read_text())
        state = result["state"]
        eligible = {b for b, g in plan["path_gate"].items() if g["eligible"]}
        if (
            not complete["success"]
            or complete["experiment"] != "envelope"
            or complete["stage"] != "path"
            or result["stage"] != "path"
            or complete["state"] != state
            or state in states
            or complete["rows"] != len(eligible)
            or len(result["rows"]) != len(eligible)
            or {b["block"] for b in result["rows"]} != eligible
            or complete["data_manifest_sha256"] != result["data_manifest_sha256"]
            or frozen != plan
        ):
            raise ValueError("Incomplete or inconsistent path evidence")
        states.add(state)
        completions.append(complete)
        for block in result["rows"]:
            label = block["block"]
            if block["baseline_source"] != inputs["blocks"][state][label]:
                raise ValueError("Path baseline provenance changed")
            points = block["rows"]
            if len(points) != 5 or {(r["alpha"], r["fraction"]) for r in points} != expected_points:
                raise ValueError("Missing or duplicate path points")
            for row in points:
                h = np.asarray(row["projected_hessian"], dtype=np.float64)
                if h.shape != (4, 4) or not np.isfinite(h).all():
                    raise ValueError("Invalid fixed-anchor Hessian")
                symmetry = float(np.linalg.norm(h - h.T) / max(np.linalg.norm(h), 1e-30))
                if symmetry >= 1e-4 or not np.isclose(symmetry, row["symmetry_relative"], atol=1e-12):
                    raise ValueError("Unvalidated path symmetry")
                checks = row["derivative_checks"]
                if row["fraction"] == 0:
                    if not row["reused_test_responses"] or checks:
                        raise ValueError("Starting response was not reused as specified")
                else:
                    if len(checks) != 2 or {c["direction"] for c in checks} != {0, 3}:
                        raise ValueError("Missing path derivative checks")
                    for check in checks:
                        last = check["trials"][-2:]
                        if not check["passed"] or len(last) != 2 or not all(t["relative_error"] < 0.02 for t in last):
                            raise ValueError("Path finite differences failed")
                if not np.isclose(row["length"], row["fraction"] * block["length"], atol=1e-12):
                    raise ValueError("Powers do not share the prescribed physical length")
                expected_remainder = row["loss_change"] + row["length"] * row["initial_alignment"]
                if not np.isclose(row["first_order_remainder"], expected_remainder, atol=1e-12):
                    raise ValueError("Inconsistent loss remainder")
                envelopes = row["envelopes"]
                if (
                    len(envelopes) != 6
                    or {(e["gradient_reference"], e["control"]) for e in envelopes} != expected_controls
                ):
                    raise ValueError("Missing or duplicate path envelopes")
                identity = dict(
                    state=state, block=label, point=row["point"], alpha=row["alpha"], fraction=row["fraction"]
                )
                losses.append(
                    dict(
                        **identity,
                        length=row["length"],
                        actual_norm=row["actual_norm"],
                        loss_change=row["loss_change"],
                        first_order_remainder=row["first_order_remainder"],
                        symmetry_relative=symmetry,
                        measurement_seconds=row["native_seconds"],
                    )
                )
                allowance = 0.02 * np.linalg.norm(h, "fro") + 0.5 * np.linalg.norm(h - h.T, 2)
                for envelope in envelopes:
                    c = np.asarray(envelope["projected_envelope"], dtype=np.float64)
                    if c.shape != (4, 4) or not np.isfinite(c).all() or np.linalg.eigvalsh(c).min() <= 0:
                        raise ValueError("Invalid path envelope")
                    rho = normalized_hessian_norm(h, c)
                    if not np.isclose(rho, envelope["normalized_hessian_norm"], rtol=1e-9, atol=1e-12):
                        raise ValueError("Saved coverage score does not reproduce")
                    margin = float(allowance / np.linalg.eigvalsh(c).min())
                    measurements.append(
                        dict(
                            **identity,
                            gradient_reference=envelope["gradient_reference"],
                            control=envelope["control"],
                            normalized_hessian_norm=rho,
                            nominal_violation=bool(rho > 1),
                            diagnostic_margin=margin,
                            exceeds_diagnostic_margin=bool(rho > 1 + margin),
                            mean_trace=float(np.trace(c) / 4),
                        )
                    )
    if states != {"eighth-middle", "quarter-middle"}:
        raise ValueError("Require both prespecified checkpoint paths")
    return dict(
        complete=completions,
        plan_sha256=digest,
        rows=measurements,
        losses=losses,
        nominal_violations=sum(r["nominal_violation"] for r in measurements),
        violations_beyond_diagnostic_margin=sum(r["exceeds_diagnostic_margin"] for r in measurements),
        scope=(
            "Original frozen coefficients on four fixed anchors at five sampled points per block; "
            "no uniform path guarantee"
        ),
        loss_scope="Reference-panel loss remainders for block moves; not held-out full-model optimizer performance",
        numerical_scope=(
            "2% Frobenius sensitivity allowance plus asymmetry is diagnostic, not a uniform error certificate"
        ),
        timing_scope=(
            "Measurement time includes gradients, HVPs, finite differences and losses; not native optimizer throughput"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, nargs="+", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_paths(args.paths, json.loads(args.plan.read_text()), json.loads(args.inputs.read_text()))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
