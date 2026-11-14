# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np

from experiments.ema_gram.analyze_geometry import paired_interval
from experiments.ema_gram.envelope_path import measure_path


def test_paired_interval_preserves_constant_difference_with_unequal_weights():
    interval = paired_interval(np.full((8, 16), 0.25), np.arange(1, 17), draws=1000)
    np.testing.assert_allclose([interval[k] for k in ("mean", "lower", "upper")], 0.25, atol=1e-12)


def test_path_quadratic_remainders_and_fixed_anchors_match_exact_calculation():
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
