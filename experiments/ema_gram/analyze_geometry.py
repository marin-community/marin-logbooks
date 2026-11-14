# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate completed geometry artifacts and summarize paired held-out decisions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.ema_gram.fit_geometry_envelopes import components
from experiments.ema_gram.geometry_math import normalized_hessian_norm


def paired_interval(values, weights, seed=202611135, draws=10000):
    """Paired two-way bootstrap: independent update draws and shared text groups."""
    values, weights = np.asarray(values, np.float64), np.asarray(weights, np.float64)
    rng = np.random.default_rng(seed)
    n, groups = values.shape
    update_indices = rng.integers(n, size=(draws, n))
    text_indices = rng.integers(groups, size=(draws, groups))
    sampled = values[update_indices[:, :, None], text_indices[:, None, :]]
    sampled_weights = weights[text_indices]
    estimates = np.mean(
        np.sum(sampled * sampled_weights[:, None, :], axis=2) / np.sum(sampled_weights, axis=1)[:, None], axis=1
    )
    return dict(
        mean=float(np.mean(np.average(values, axis=1, weights=weights))),
        lower=float(np.quantile(estimates, 0.025)),
        upper=float(np.quantile(estimates, 0.975)),
        bootstrap_samples=draws,
        seed=seed,
        scope=(
            "Conditional on this checkpoint, calibration panel and fitted selection policies; "
            "paired update-draw and evaluation-group resampling, not training-seed uncertainty"
        ),
    )


def analyze_power(directory):
    complete = json.loads((directory / "completed.json").read_text())
    result = json.loads((directory / "power-results.json").read_text())
    locked = json.loads((directory / "locked-predictions.json").read_text())
    if not complete["success"] or complete["experiment"] != "power" or complete["rows"] != 24:
        raise ValueError("Power study incomplete")
    digest = hashlib.sha256(json.dumps(locked, sort_keys=True).encode()).hexdigest()
    if digest != result["locked_predictions_sha256"]:
        raise ValueError("Locked prediction digest mismatch")
    if complete["data_manifest_sha256"] != result["data_manifest_sha256"]:
        raise ValueError("Data identity mismatch")
    if result["choices"] != locked["choices"]:
        raise ValueError("Selector choices changed after locking")
    rows = {(r["draw"], r["alpha"]): r for r in result["rows"]}
    if set(rows) != {(d, a) for d in range(8) for a in (0.125, 0.1875, 0.25)}:
        raise ValueError("Missing or duplicate power candidates")
    for row, frozen in zip(result["rows"], locked["rows"], strict=True):
        if {k: v for k, v in row.items() if k != "evaluation"} != frozen:
            raise ValueError("Calibration evidence changed after locking")
        if row["alpha"] == 0.1875 and not row["alpha_derivative"]["passed"]:
            raise ValueError("Unvalidated power tangent")
        if len(row["evaluation"]["direct"]["losses"]) != 16:
            raise ValueError("Evaluation panel incomplete")
    baseline = np.asarray(result["evaluation_baseline"])
    weights = result["evaluation_weights"]
    policies = ("fixed_eighth", "fixed_quarter", "alignment", "quadratic", "calibration_best")
    selected, comparisons = {}, {}
    for length_rule in ("direct", "quadratic", "equal", "native"):
        curves = {}
        for policy in policies:
            curves[policy] = np.stack(
                [
                    np.asarray(rows[c["draw"], c[policy]]["evaluation"][length_rule]["losses"]) - baseline
                    for c in result["choices"]
                ]
            )
        selected[length_rule] = {p: paired_interval(v, weights) for p, v in curves.items()}
        comparisons[length_rule] = {}
        for first, second in (
            ("quadratic", "alignment"),
            ("quadratic", "fixed_eighth"),
            ("quadratic", "fixed_quarter"),
            ("fixed_quarter", "fixed_eighth"),
        ):
            comparisons[length_rule][first + "_minus_" + second] = paired_interval(
                curves[first] - curves[second], weights
            )
    return dict(
        state=result["state"],
        complete=complete,
        selected_loss_changes=selected,
        paired_loss_differences=comparisons,
        calibration_boundary_winners=sum(r["grid_boundary_winner"] for r in rows.values()),
        selected_powers={p: [c[p] for c in result["choices"]] for p in policies},
        sign_convention="Negative loss change or paired difference is better",
        independence="Eight update batches are repeated measurements at one checkpoint, not eight training seeds",
    )


def analyze_muon(directory):
    complete = json.loads((directory / "completed.json").read_text())
    result = json.loads((directory / "muon-results.json").read_text())
    locked = json.loads((directory / "locked-predictions.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    variants = protocol["variants"]
    expected = {(d, v) for d in range(4) for v in variants}
    rows = {(r["draw"], r["variant"]): r for r in result["rows"]}
    if not complete["success"] or complete["experiment"] != "muon" or complete["rows"] != 24:
        raise ValueError("Muon study incomplete")
    if set(rows) != expected or len(result["rows"]) != len(expected):
        raise ValueError("Muon candidates missing or duplicated")
    digest = hashlib.sha256(json.dumps(locked, sort_keys=True).encode()).hexdigest()
    if digest != result["locked_predictions_sha256"] or result["choices"] != locked["choices"]:
        raise ValueError("Muon choices changed after locking")
    if complete["data_manifest_sha256"] != result["data_manifest_sha256"]:
        raise ValueError("Muon data identity mismatch")
    if not result["HVP_validation"]["passed"] or result["HVP_symmetry_relative_error"] >= 1e-4:
        raise ValueError("Unvalidated Muon reference Hessian")
    if len(result["derivative_checks"]) != 4 or not all(c["passed"] for c in result["derivative_checks"]):
        raise ValueError("Unvalidated Muon F5 derivative")
    for row, frozen in zip(result["rows"], locked["rows"], strict=True):
        if {k: v for k, v in row.items() if k != "evaluation"} != frozen:
            raise ValueError("Muon calibration evidence changed after locking")
        if len(row["evaluation"]["direct"]["losses"]) != 16:
            raise ValueError("Muon evaluation panel incomplete")
    baseline, weights = np.asarray(result["evaluation_baseline"]), result["evaluation_weights"]
    selected, comparisons = {}, {}
    for rule in ("direct", "equal", "quadratic", "reference_native"):
        changes = {
            v: np.stack([np.asarray(rows[d, v]["evaluation"][rule]["losses"]) - baseline for d in range(4)])
            for v in variants
        }
        selected[rule] = {v: paired_interval(values, weights) for v, values in changes.items()}
        comparisons[rule] = {
            v: paired_interval(values - changes["muon"], weights) for v, values in changes.items() if v != "muon"
        }
    return dict(
        state=result["state"],
        complete=complete,
        selected_loss_changes=selected,
        paired_vs_plain_muon=comparisons,
        reference_learning_rate=result["reference_learning_rate"],
        reference_adam_learning_rate=result["reference_adam_learning_rate"],
        boundary_winners=sum(r["grid_boundary_winner"] for r in rows.values()),
        numerical_diagnostics=result["derivative_checks"],
        independence="Four update draws per checkpoint; no independent training-seed replication",
        sign_convention="Negative loss difference means improvement over plain Muon",
    )


def analyze_history(directory):
    complete = json.loads((directory / "completed.json").read_text())
    result = json.loads((directory / "history-results.json").read_text())
    locked = json.loads((directory / "locked-predictions.json").read_text())
    if not complete["success"] or complete["experiment"] != "history" or complete["rows"] != 160:
        raise ValueError("History comparison incomplete")
    digest = hashlib.sha256(json.dumps(locked, sort_keys=True).encode()).hexdigest()
    if digest != result["locked_predictions_sha256"]:
        raise ValueError("History prediction lock changed")
    if complete["data_manifest_sha256"] != result["data_manifest_sha256"]:
        raise ValueError("History panel identity mismatch")
    if not result["HVP_validation"]["passed"] or result["HVP_symmetry_relative_error"] >= 1e-4:
        raise ValueError("History reference Hessian is unvalidated")
    names = [c["name"] for c in result["conditions"]]
    rows = {(r["draw"], r["condition"], r["alpha"]): r for r in result["rows"]}
    expected = {(d, c, a) for d in range(8) for c in names for a in (0.125, 0.25)}
    if len(names) != 10 or len(result["rows"]) != 160 or set(rows) != expected:
        raise ValueError("Missing history candidate")
    for row, frozen in zip(result["rows"], locked["rows"], strict=True):
        if {k: v for k, v in row.items() if k != "evaluation"} != frozen:
            raise ValueError("History calibration changed after locking")
        if len(row["evaluation"]["direct"]["losses"]) != 16:
            raise ValueError("History evaluation incomplete")
    power_differences, history_effects, reversals = {}, {}, []
    weights = result["evaluation_weights"]
    for rule in ("equal", "direct"):
        paired = {
            c: np.stack(
                [
                    np.asarray(rows[d, c, 0.25]["evaluation"][rule]["losses"])
                    - np.asarray(rows[d, c, 0.125]["evaluation"][rule]["losses"])
                    for d in range(8)
                ]
            )
            for c in names
        }
        power_differences[rule] = {c: paired_interval(v, weights) for c, v in paired.items()}
        history_effects[rule] = {
            c: paired_interval(v - paired["saved"], weights) for c, v in paired.items() if c != "saved"
        }
        saved_mean = power_differences[rule]["saved"]["mean"]
        for condition in names:
            interval = power_differences[rule][condition]
            if interval["mean"] * saved_mean < 0:
                reversals.append(
                    dict(
                        rule=rule,
                        condition=condition,
                        saved_difference=saved_mean,
                        changed_difference=interval["mean"],
                        changed_interval_excludes_zero=interval["lower"] > 0 or interval["upper"] < 0,
                    )
                )
    return dict(
        state=result["state"],
        complete=complete,
        conditions=result["conditions"],
        quarter_minus_eighth=power_differences,
        change_from_saved=history_effects,
        nominal_ranking_reversals=reversals,
        matched_replay_needed=bool(reversals),
        replay_scope="A reversal triggers matched moving-versus-frozen replay; it does not establish temporal lag",
        independence="Conditional checkpoint uncertainty; fresh histories and update draws are not training seeds",
    )


def analyze_envelope(directories, plan):
    measured = []
    path_inputs = dict(plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(), blocks={})
    for directory in directories:
        complete = json.loads((directory / "completed.json").read_text())
        result = json.loads((directory / "envelope-results.json").read_text())
        frozen = json.loads((directory / "frozen-envelope-plan.json").read_text())
        if not complete["success"] or complete["stage"] != "test" or complete["rows"] != 2 or frozen != plan:
            raise ValueError("Test artifacts do not match the complete frozen fit")
        path_inputs["blocks"][result["state"]] = {}
        for row in result["rows"]:
            if row["identities"] != plan["inputs"][result["state"]][row["block"]]:
                raise ValueError("Test basis or gradient changed")
            if row["symmetry_relative"] >= 1e-4 or not all(c["passed"] for c in row["derivative_checks"]):
                raise ValueError("Unvalidated test Hessian")
            path_inputs["blocks"][result["state"]][row["block"]] = dict(
                identities=row["identities"],
                arrays=row["arrays"],
                completed=complete,
                result_sha256=hashlib.sha256((directory / "envelope-results.json").read_bytes()).hexdigest(),
            )
            h = np.asarray(row["projected_hessian"])
            # This is a sensitivity allowance, not an error certificate for untested directions.
            absolute_allowance = 0.02 * np.linalg.norm(h, "fro") + 0.5 * np.linalg.norm(h - h.T, 2)
            for control, fit in plan["fits"][row["block"]].items():
                envelope = np.einsum("i,ijk->jk", fit["coefficients"], components(row, control))
                rho = normalized_hessian_norm(h, envelope)
                margin = absolute_allowance / np.linalg.eigvalsh(envelope).min()
                measured.append(
                    dict(
                        state=result["state"],
                        block=row["block"],
                        control=control,
                        normalized_hessian_norm=rho,
                        nominal_violation=rho > 1,
                        diagnostic_margin=float(margin),
                        exceeds_diagnostic_margin=rho > 1 + margin,
                        mean_trace=float(np.trace(envelope) / len(h)),
                        numerical_scope=(
                            "2% Frobenius-Hessian sensitivity allowance plus measured asymmetry; "
                            "finite differences check two anchors, so this is not a uniform error certificate"
                        ),
                    )
                )
    if {r["state"] for r in measured} != {"eighth-middle", "quarter-middle"} or len(measured) != 12:
        raise ValueError("Require the two complete prespecified test screens")
    matched = []
    for block in ("attention", "mlp"):
        for control in ("scalar", "oriented", "rotated"):
            group = [r for r in measured if (r["block"], r["control"]) == (block, control)]
            inflation = max(1.0, max(r["normalized_hessian_norm"] for r in group))
            matched.append(
                dict(
                    block=block,
                    control=control,
                    inflation=inflation,
                    mean_trace=float(inflation * np.mean([r["mean_trace"] for r in group])),
                    scope=(
                        "Post-hoc common inflation across both test states for matched nominal coverage; "
                        "diagnostic only, not an independently validated bound"
                    ),
                )
            )
    return dict(
        rows=measured,
        path_inputs=path_inputs,
        matched_coverage=matched,
        scope="Sampled block subspaces at two fixed checkpoints; not a uniform theorem or full-network bound",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power", type=Path, nargs="*", default=[])
    parser.add_argument("--history", type=Path, nargs="*", default=[])
    parser.add_argument("--muon", type=Path, nargs="*", default=[])
    parser.add_argument("--envelope-test", type=Path, nargs="*", default=[])
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = dict(power=[analyze_power(p) for p in args.power], muon=[analyze_muon(p) for p in args.muon])
    output["history"] = [analyze_history(p) for p in args.history]
    if args.envelope_test:
        output["envelope"] = analyze_envelope(args.envelope_test, json.loads(args.plan.read_text()))
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
