# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-held-out envelopes and direct-evaluation plans for curvature probes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack

from experiments.ema_gram.gradient_curvature_math import paired_interval

CANONICAL_FRACTIONS = (0, 0.05, 0.1, 0.15, 0.3, 0.5, 1.0)
RULES = ("constant", "linear", "quadratic")


def fit_envelope(rows, power, quantile=0.9):
    """Nonnegative quantile regression, with equal total weight per checkpoint."""
    g = np.asarray([r["G"] for r in rows])
    y = np.asarray([r["Ldir"] for r in rows])
    scale = float(np.median(g))
    yscale = float(np.median(y))
    if scale <= 0 or yscale <= 0 or not np.isfinite(g).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite or degenerate curvature data")
    features = np.ones((len(rows), 1)) if power == 0 else np.column_stack((np.ones(len(rows)), (g / scale) ** power))
    counts = {name: sum(r["state"] == name for r in rows) for name in {r["state"] for r in rows}}
    weights = np.asarray([1 / counts[r["state"]] / len(counts) for r in rows])
    # y - X beta = positive residual - negative residual.
    equality = hstack((csr_matrix(features), eye(len(rows)), -eye(len(rows))), format="csr")
    objective = np.r_[np.zeros(features.shape[1]), quantile * weights, (1 - quantile) * weights]
    solution = linprog(objective, A_eq=equality, b_eq=y / yscale, bounds=(0, None), method="highs")
    if not solution.success:
        raise ValueError(f"Quantile regression failed: {solution.message}")
    beta = solution.x[: features.shape[1]] * yscale
    return dict(
        power=power,
        intercept=float(beta[0]),
        coefficient=float(beta[1] / scale**power) if power else 0.0,
        quantile=quantile,
        training_states=sorted(counts),
    )


def envelope(model, g):
    return model["intercept"] + model["coefficient"] * np.asarray(g) ** model["power"]


def canonical_rows(results):
    return [
        dict(state=result["state"], **r)
        for result in results
        for r in result["rows"]
        if r["mode"] == "full" and r["groups"] == 8 and r["fraction"] in CANONICAL_FRACTIONS
    ]


def checkpoint_summaries(results):
    summaries = {}
    for result in results:
        initial = [r for r in result["rows"] if r["mode"] == "full" and r["groups"] == 8 and r["fraction"] == 0]
        g = np.asarray([r["G"] for r in initial])
        if len(initial) != 8 or not np.allclose(g, g[0], rtol=1e-5, atol=1e-7):
            raise ValueError("Expected four paired draws and one common starting gradient")
        summaries[result["state"]] = dict(
            counter=result["counter"],
            trajectory=result["trajectory"],
            G=float(g[0]),
            data_manifest_sha256=result["data_manifest_sha256"],
            reference_length=float(np.median([r["reference_length"] for r in initial])),
        )
    return summaries


def heldout_fits(results):
    rows = canonical_rows(results)
    folds = {}
    for heldout in sorted({r["state"] for r in rows}):
        train = [r for r in rows if r["state"] != heldout]
        test = [r for r in rows if r["state"] == heldout]
        models = {}
        for power, name in enumerate(RULES):
            model = fit_envelope(train, power)
            truth = np.asarray([r["Ldir"] for r in test])
            prediction = envelope(model, [r["G"] for r in test])
            residual = truth - prediction
            models[name] = dict(
                **model,
                heldout_coverage=float(np.mean(residual <= 1e-8)),
                heldout_mae=float(np.mean(abs(residual))),
                heldout_pinball=float(np.mean(np.maximum(0.9 * residual, -0.1 * residual))),
                heldout_count=len(test),
            )
        folds[heldout] = models
    return folds


def calibration_plan(results, fits, source_hashes):
    """Exclude the held-out checkpoint entirely from scalar calibration tasks."""
    states = checkpoint_summaries(results)
    tasks = {name: [] for name in states}
    folds = {}
    for heldout, models in fits.items():
        train = [name for name in states if name != heldout]
        reference = float(np.median([states[name]["reference_length"] for name in train]))
        caps = [0.01 * reference, 3.0 * reference]
        rules = {}
        for name in RULES:
            model = models[name]
            rates = [
                1.0 if name == "constant" else states[s]["G"] / float(envelope(model, 2 * states[s]["G"])) for s in train
            ]
            if not np.isfinite(rates).all() or min(rates) <= 0:
                raise ValueError("Degenerate adaptive rule")
            scalars = [fraction * reference / float(np.median(rates)) for fraction in (0.1, 0.3, 1.0)]
            rules[name] = dict(model=model, scalars=scalars)
            for state, rate in zip(train, rates, strict=True):
                for candidate, scalar in enumerate(scalars):
                    proposed = scalar * rate
                    tasks[state].append(
                        dict(
                            heldout=heldout,
                            rule=name,
                            candidate=candidate,
                            scalar=scalar,
                            proposed_length=proposed,
                            length=float(np.clip(proposed, *caps)),
                            caps=caps,
                        )
                    )
        folds[heldout] = dict(training_states=train, caps=caps, reference_length=reference, rules=rules)
    return dict(
        stage="calibration",
        states=states,
        folds=folds,
        tasks=tasks,
        source_hashes=source_hashes,
        selection="Mean direct loss change, equal weight per checkpoint, power, and update draw",
    )


def heldout_evaluation_plan(calibration, measurements):
    """Select scalars using calibration states only, then emit held-out tasks."""
    by_state = {r["state"]: r for r in measurements}
    plan_hash = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    if set(by_state) != set(calibration["states"]):
        raise ValueError("Need completed direct calibration measurements for every checkpoint")
    if any(r["plan_hash"] != plan_hash for r in measurements):
        raise ValueError("Calibration outputs came from a different plan")
    tasks = {name: [] for name in by_state}
    selections = {}
    for heldout, fold in calibration["folds"].items():
        selections[heldout] = {}
        for rule, info in fold["rules"].items():
            scores = []
            for candidate in range(len(info["scalars"])):
                state_scores = []
                for state in fold["training_states"]:
                    selected = [
                        r
                        for r in by_state[state]["rows"]
                        if r["heldout"] == heldout and r["rule"] == rule and r["candidate"] == candidate
                    ]
                    if len(selected) != 8 or {(r["draw"], r["alpha"]) for r in selected} != {
                        (d, a) for d in range(4) for a in (0.125, 0.25)
                    }:
                        raise ValueError("Missing or duplicated paired calibration measurements")
                    state_scores.append(np.mean([r["loss_change"] for r in selected]))
                scores.append(float(np.mean(state_scores)))
            chosen = int(np.argmin(scores))
            scalar = info["scalars"][chosen]
            g = calibration["states"][heldout]["G"]
            rate = 1.0 if rule == "constant" else g / float(envelope(info["model"], 2 * g))
            proposed = scalar * rate
            tasks[heldout].append(
                dict(
                    heldout=heldout,
                    rule=rule,
                    candidate=chosen,
                    scalar=scalar,
                    proposed_length=proposed,
                    length=float(np.clip(proposed, *fold["caps"])),
                    caps=fold["caps"],
                )
            )
            selections[heldout][rule] = dict(candidate=chosen, scalar=scalar, calibration_scores=scores)
    return dict(
        stage="heldout",
        states=calibration["states"],
        tasks=tasks,
        selections=selections,
        calibration_plan_hash=plan_hash,
        source_hashes=calibration["source_hashes"],
    )


def paired_summaries(results):
    summaries = []
    for result in results:
        for mode in sorted({r["mode"] for r in result["rows"]}):
            for count in sorted({r["groups"] for r in result["rows"]}):
                for fraction in (0.1, 0.3, 1.0):
                    rows = {
                        (r["draw"], r["alpha"]): r
                        for r in result["rows"]
                        if r["mode"] == mode and r["groups"] == count and r["fraction"] == fraction
                    }
                    differences = [np.asarray(rows[d, 0.125]["losses"]) - rows[d, 0.25]["losses"] for d in range(4)]
                    summaries.append(
                        dict(
                            state=result["state"],
                            mode=mode,
                            groups=count,
                            fraction=fraction,
                            **paired_interval(differences, result["reference_weights"][:count]),
                        )
                    )
    return summaries


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows
        )


def adaptive_summaries(plan, results):
    """Analyze only complete direct evaluations of the selected held-out plan."""
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    if plan["stage"] != "heldout" or len(results) != len(plan["states"]):
        raise ValueError("Need one held-out result for every planned checkpoint")
    if {r["state"] for r in results} != set(plan["states"]):
        raise ValueError("Held-out checkpoint set differs from the plan")
    paired, changes, versus_constant = [], [], []
    for result in results:
        state = result["state"]
        if result["stage"] != "heldout" or result["plan_hash"] != plan_hash:
            raise ValueError("Result is not from the selected held-out plan")
        expected = {(rule, d, a) for rule in RULES for d in range(4) for a in (0.125, 0.25)}
        rows = {(r["rule"], r["draw"], r["alpha"]): r for r in result["rows"]}
        if len(result["rows"]) != len(expected) or set(rows) != expected:
            raise ValueError("Missing or duplicated held-out evaluations")
        for task in plan["tasks"][state]:
            for d in range(4):
                for a in (0.125, 0.25):
                    row = rows[task["rule"], d, a]
                    if any(row[key] != value for key, value in task.items()):
                        raise ValueError("Evaluation parameters differ from the selected task")
        weights = result["reference_weights"]
        baseline = np.asarray(result["baseline"])
        for rule in RULES:
            differences = [np.asarray(rows[rule, d, 0.125]["losses"]) - rows[rule, d, 0.25]["losses"] for d in range(4)]
            paired.append(dict(state=state, rule=rule, **paired_interval(differences, weights)))
            for a in (0.125, 0.25):
                selected = [rows[rule, d, a] for d in range(4)]
                loss_changes = [np.asarray(row["losses"]) - baseline for row in selected]
                changes.append(
                    dict(
                        state=state,
                        rule=rule,
                        alpha=a,
                        **paired_interval(loss_changes, weights),
                        length=selected[0]["length"],
                        proposed_length=selected[0]["proposed_length"],
                        cap_active=selected[0]["length"] != selected[0]["proposed_length"],
                        nonpositive_alignment_count=sum(row["nonpositive_alignment"] for row in selected),
                    )
                )
                if rule != "constant":
                    differences = [
                        np.asarray(rows[rule, d, a]["losses"]) - rows["constant", d, a]["losses"] for d in range(4)
                    ]
                    versus_constant.append(
                        dict(state=state, rule=rule, alpha=a, **paired_interval(differences, weights))
                    )
    return dict(
        plan_hash=plan_hash,
        paired=paired,
        loss_changes=changes,
        versus_constant=versus_constant,
        interval_scope="Pointwise 95% crossed-bootstrap intervals conditional on each checkpoint and selected plan; "
        "no calibration-selection or training-seed uncertainty, and no multiplicity correction",
    )


def paired_decomposition(results):
    """Separate alignment, initial curvature, and path growth without fitting losses."""
    rows = []
    for result in results:
        endpoints = {(r["mode"], r["draw"], r["alpha"], r["fraction"]): r for r in result["rows"] if r["groups"] == 8}
        initial = {
            (r["mode"], r["draw"], r["alpha"]): r for r in result["rows"] if r["fraction"] == 0 and r["groups"] == 8
        }
        for comparison in result["comparisons"]:
            eighth = initial[comparison["mode"], comparison["draw"], 0.125]
            quarter = initial[comparison["mode"], comparison["draw"], 0.25]
            length = eighth["reference_length"] * comparison["fraction"]
            alignment = -length * (eighth["alignment"] - quarter["alignment"])
            initial_curvature = 0.5 * length**2 * (eighth["kappa"] - quarter["kappa"])
            predicted = comparison["predicted_eighth_minus_quarter"]
            observed_changes = {
                str(a): endpoints[comparison["mode"], comparison["draw"], a, comparison["fraction"]]["loss_change"]
                for a in (0.125, 0.25)
            }
            rows.append(
                dict(
                    state=result["state"],
                    **comparison,
                    length=length,
                    alignment_contribution=alignment,
                    initial_curvature_contribution=initial_curvature,
                    path_growth_contribution=predicted - alignment - initial_curvature,
                    remaining_prediction_residual=comparison["observed_eighth_minus_quarter"] - predicted,
                    observed_changes=observed_changes,
                    individual_prediction_errors={
                        a: comparison["predicted_changes"][a] - value for a, value in observed_changes.items()
                    },
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    plans = parser.add_mutually_exclusive_group()
    plans.add_argument("--calibration-plan", type=Path)
    plans.add_argument("--heldout-plan", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [json.loads(path.read_text()) for path in args.inputs]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.inputs}
    manifest_hashes = set()
    for result, path in zip(results, args.inputs, strict=True):
        completion = json.loads((path.parent / "completed.json").read_text())
        if not completion["success"] or completion["state"] != result["state"]:
            raise ValueError("Every result needs its matching successful completion record")
        manifest_hash = hashlib.sha256((path.parent / "data-manifest.json").read_bytes()).hexdigest()
        result["data_manifest_sha256"] = manifest_hash
        manifest_hashes.add(manifest_hash)
    if len(manifest_hashes) != 1:
        raise ValueError("Reference/update panels differ across checkpoints")
    if args.heldout_plan:
        analysis = adaptive_summaries(json.loads(args.heldout_plan.read_text()), results)
        analysis["source_hashes"] = hashes
        for key in ("paired", "loss_changes", "versus_constant"):
            write_csv(args.output / f"adaptive-{key}.csv", analysis[key])
        (args.output / "adaptive-analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
        return
    if args.calibration_plan:
        plan = heldout_evaluation_plan(json.loads(args.calibration_plan.read_text()), results)
        (args.output / "heldout-plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
        return
    if len({r["state"] for r in results}) != len(results):
        raise ValueError("Only one complete result per checkpoint is permitted")
    for result in results:
        if not result["derivative_checks"] or not all(x["passed"] for x in result["derivative_checks"]):
            raise ValueError("Numerical validation must pass before analysis")
        checkpoint_summaries([result])
        expected = 24 if any(r["mode"] == "matrix_only" for r in result["rows"]) else 12
        if len(result["comparisons"]) != expected:
            raise ValueError("Path measurements are incomplete")
    rows = [
        dict(state=r["state"], trajectory=r["trajectory"], checkpoint=r["checkpoint"], **row)
        for r in results
        for row in r["rows"]
    ]
    paired = paired_summaries(results)
    write_csv(args.output / "path-measurements.csv", rows)
    write_csv(args.output / "paired-common-length.csv", paired)
    decomposition = paired_decomposition(results)
    write_csv(args.output / "paired-loss-decomposition.csv", decomposition)
    analysis = dict(
        source_hashes=hashes,
        paired=paired,
        decomposition=decomposition,
        canonical_fractions=CANONICAL_FRACTIONS,
        interval_scope="Conditional on checkpoints; shared groups and paired draws are resampled; no seed variation",
    )
    if len(results) == 6:
        fits = heldout_fits(results)
        analysis["fits"] = fits
        plan = calibration_plan(results, fits, hashes)
        (args.output / "calibration-plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    else:
        analysis["fit_status"] = "Waiting for all six checkpoints; no held-out fit or adaptive calibration yet"
    (args.output / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
