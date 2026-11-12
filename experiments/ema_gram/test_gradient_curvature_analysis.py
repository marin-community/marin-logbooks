# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Numerical model recovery and checkpoint-exclusion regressions."""

import hashlib
import json

import numpy as np

from experiments.ema_gram.gradient_curvature_analysis import (
    calibration_plan,
    envelope,
    fit_envelope,
    heldout_evaluation_plan,
    heldout_fits,
)


def test_quantile_envelope_recovers_known_growth():
    rows = [dict(state=str(i % 6), G=g, Ldir=2 + 3 * g**2) for i, g in enumerate(np.linspace(0.1, 4, 60))]
    fitted = fit_envelope(rows, 2)
    np.testing.assert_allclose(envelope(fitted, [0.5, 2, 5]), 2 + 3 * np.asarray([0.5, 2, 5]) ** 2, rtol=1e-6)
    constant = fit_envelope(rows, 0)
    coverage = np.mean([r["Ldir"] <= envelope(constant, r["G"]) + 1e-8 for r in rows])
    assert 0.88 <= coverage <= 0.92


def test_heldout_checkpoint_cannot_select_its_scalar():
    results = []
    for state in range(6):
        rows = [
            dict(
                mode="full",
                groups=8,
                fraction=f,
                draw=d,
                alpha=a,
                G=state + 1 + f,
                Ldir=2 + 3 * (state + 1 + f) ** 2,
                reference_length=state + 1,
            )
            for f in (0, 0.05, 0.1, 0.15, 0.3, 0.5, 1.0)
            for d in range(4)
            for a in (0.125, 0.25)
        ]
        results.append(
            dict(state=str(state), counter=state, trajectory=0.125, rows=rows, data_manifest_sha256="unit-test")
        )
    fits = heldout_fits(results)
    plan = calibration_plan(results, fits, {})
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    measurements = [
        dict(
            state=state,
            plan_hash=plan_hash,
            rows=[
                dict(**t, draw=d, alpha=a, loss_change=(t["candidate"] - 1) ** 2 + 0.01 * d)
                for t in tasks
                for d in range(4)
                for a in (0.125, 0.25)
            ],
        )
        for state, tasks in plan["tasks"].items()
    ]
    selected = heldout_evaluation_plan(plan, measurements)
    assert all(t["candidate"] == 1 for tasks in selected["tasks"].values() for t in tasks)
    for row in measurements[0]["rows"]:
        row["loss_change"] = -1e9 * row["candidate"]
    changed = heldout_evaluation_plan(plan, measurements)
    assert changed["selections"]["0"] == selected["selections"]["0"]
    assert changed["selections"]["1"] != selected["selections"]["1"]
    for row in results[0]["rows"]:
        row["Ldir"] *= 1000
    refitted = heldout_fits(results)
    for rule in ("constant", "linear", "quadratic"):
        assert refitted["0"][rule]["coefficient"] == fits["0"][rule]["coefficient"]
        assert refitted["0"][rule]["intercept"] == fits["0"][rule]["intercept"]
