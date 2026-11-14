# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.ema_gram.fit_geometry_envelopes import components, fit_envelope
from experiments.ema_gram.geometry_math import envelope_components, normalized_hessian_norm, rotated_gradient


def test_shared_envelope_recovers_anisotropic_bound_and_catches_offdiagonal_violation():
    basis = np.eye(4).reshape(4, 2, 2)
    rows = []
    for gradient in (np.diag([3.0, 1.0]), np.array([[2.0, 0.5], [0.2, 1.0]])):
        matrices = envelope_components(gradient, basis)
        hessian = np.einsum("i,ijk->jk", [0.2, 0.7, 0.3], matrices)
        rows.append(
            dict(
                projected_hessian=hessian,
                envelope_components=matrices,
                rotated_components=envelope_components(rotated_gradient(gradient, 72), basis),
                gradient_norm=np.linalg.norm(gradient),
            )
        )
    fits = {control: fit_envelope(rows, control) for control in ("scalar", "oriented", "rotated")}
    np.testing.assert_allclose(fits["oriented"]["coefficients"], [0.2, 0.7, 0.3], atol=1e-5)
    assert fits["oriented"]["calibration_mean_trace"] < 0.8 * fits["scalar"]["calibration_mean_trace"]
    for control, fit in fits.items():
        assert min(fit["coefficients"]) >= 0 and fit["coefficients"][0] > 0
        for row in rows:
            envelope = np.einsum("i,ijk->jk", fit["coefficients"], components(row, control))
            assert normalized_hessian_norm(row["projected_hessian"], envelope) <= 1 + 1e-7
    hessian = np.array([[0.8, 0.4], [0.4, 0.8]])
    assert np.max(np.abs(np.diag(hessian))) < 1
    assert normalized_hessian_norm(hessian, np.eye(2)) > 1.19
