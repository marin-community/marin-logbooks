# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.ema_gram.analyze_envelope_paths import analyze_paths
from experiments.ema_gram.analyze_geometry import paired_interval, prediction_error_comparison
from experiments.ema_gram.envelope_path import measure_path


def test_paired_interval_preserves_constant_difference_with_unequal_weights():
    interval = paired_interval(np.full((8, 16), 0.25), np.arange(1, 17), draws=1000)
    np.testing.assert_allclose([interval[k] for k in ("mean", "lower", "upper")], 0.25, atol=1e-12)


def test_prediction_error_uses_weighted_panel_means_and_shared_text_resamples():
    # The weighted panel change is 0.5, although neither individual group has that value.
    changes = np.broadcast_to([-1.0, 1.0], (8, 3, 2))
    linear = np.zeros((8, 3))
    quadratic = np.full((8, 3), 0.5)
    result = prediction_error_comparison(changes, linear, quadratic, [1, 3])
    assert result["linear_mean_absolute_error"] == 0.5
    assert result["quadratic_mean_absolute_error"] == 0.0
    # One shared group resample can select the negative group twice for every candidate.
    # Independent text resampling across powers or draws would shrink this interval.
    np.testing.assert_allclose(
        [result["quadratic_minus_linear"][k] for k in ("mean", "lower", "upper")], [-0.5, -0.5, 0.5]
    )


def test_path_quadratic_remainders_and_fixed_anchors_match_exact_calculation(tmp_path):
    matrix = jnp.array([[0.2, 0.3], [0.4, 0.5]])
    hessian = np.diag([1.0, 2.0, 3.0, 4.0])
    hessian[0, 1] = hessian[1, 0] = 0.2
    gradient = (hessian @ np.asarray(matrix).ravel()).reshape(2, 2)
    basis = np.eye(4).reshape(4, 2, 2)
    directions = {0.125: gradient / np.linalg.norm(gradient), 0.25: np.ones((2, 2)) / 2}
    fits = {
        control: dict(coefficients=[5.0, 0.0] if control == "scalar" else [5.0, 0.0, 0.0])
        for control in ("scalar", "oriented", "rotated")
    }
    evaluated = []

    def hvp(point, direction):
        evaluated.append(np.asarray(point))
        return (hessian @ direction.ravel()).reshape(2, 2)

    result = measure_path(
        matrix=matrix,
        gradient=gradient,
        basis=basis,
        baseline_responses=hessian.T.reshape(4, 2, 2),
        directions=directions,
        length=0.1,
        mean_gradient=lambda p: (hessian @ np.asarray(p).ravel()).reshape(2, 2),
        mean_hvp=hvp,
        losses=lambda p: [float(np.asarray(p).ravel() @ hessian @ np.asarray(p).ravel() / 2)],
        weights=[1],
        fits=fits,
        rotation_seed=42,
        save_arrays=lambda *a, **k: {},
        progress=lambda row: None,
        label="fixture",
    )
    assert len(evaluated) == 16 and len(result["rows"]) == 5
    for row in result["rows"]:
        np.testing.assert_allclose(row["projected_hessian"], hessian, atol=1e-10)
        assert row["symmetry_relative"] < 1e-10
        assert all(check["passed"] for check in row["derivative_checks"])
        if row["alpha"] is not None:
            direction = directions[row["alpha"]].ravel()
            expected = 0.5 * row["length"] ** 2 * (direction @ hessian @ direction)
            np.testing.assert_allclose(row["first_order_remainder"], expected, atol=1e-7)

    # Validate the evidence consumer on a known coupled quadratic, then corrupt
    # a finite-difference measurement while retaining its saved success flag.
    plan = {"path_gate": {"fixture": {"eligible": True}}}
    inputs = {
        "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "blocks": {},
    }
    directories = []
    for state in ("eighth-middle", "quarter-middle"):
        directory = tmp_path / state
        directory.mkdir()
        directories.append(directory)
        inputs["blocks"][state] = {"fixture": {"identities": "fixture-baseline"}}
        result["baseline_source"] = inputs["blocks"][state]["fixture"]
        complete = dict(
            success=True, experiment="envelope", stage="path", state=state, rows=1, data_manifest_sha256="data"
        )
        output = dict(state=state, stage="path", data_manifest_sha256="data", rows=[result])
        for name, data in (("completed", complete), ("envelope-path-results", output), ("frozen-envelope-plan", plan)):
            (directory / f"{name}.json").write_text(json.dumps(data))
    summary = analyze_paths(directories, plan, inputs)
    assert summary["nominal_violations"] == 0
    assert len(summary["rows"]) == 60
    expected_rho = np.linalg.eigvalsh(hessian).max() / 5
    np.testing.assert_allclose([r["normalized_hessian_norm"] for r in summary["rows"]], expected_rho)
    output["rows"][0]["rows"][1]["derivative_checks"][0]["trials"][-1]["relative_error"] = 0.2
    (directories[-1] / "envelope-path-results.json").write_text(json.dumps(output))
    with pytest.raises(ValueError, match="finite differences failed"):
        analyze_paths(directories, plan, inputs)
