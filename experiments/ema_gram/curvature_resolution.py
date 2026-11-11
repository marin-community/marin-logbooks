# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare projected HVPs with two finite-difference scales before fitting."""

import numpy as np


def projected_resolution(curvature, references, safety_factor=10.0):
    """Apply a declared empirical relative-error screen without altering spectra."""
    curvature = np.asarray(curvature, np.float64)
    references = np.asarray(references, np.float64)
    if references.shape != (2, *curvature.shape):
        raise ValueError("Expected two finite-difference arrays with the curvature batch shape")
    if not np.isfinite(curvature).all() or not np.isfinite(references).all():
        return dict(status="unresolved", reason="Missing or nonfinite projected derivative checks")
    asymmetries = np.linalg.norm(curvature - curvature.transpose(0, 2, 1), axis=(1, 2)) / np.maximum(
        np.linalg.norm(curvature, axis=(1, 2)), 1e-300
    )
    mean_raw = curvature.mean(axis=0)
    mean = (mean_raw + mean_raw.T) / 2
    values = np.linalg.eigvalsh(mean)
    result = dict(
        method="Two finite-difference scales; diagonal congruence scaling; spectral discrepancy margin",
        warning="Empirical agreement screen, not a rigorous bound on systematic differentiation error",
        safety_factor=safety_factor,
        maximum_batch_asymmetry=float(asymmetries.max()),
        eigenvalues=values.tolist(),
    )
    if asymmetries.max() > 1e-5:
        return dict(result, status="unresolved", reason="Original per-batch symmetry gate failed")
    if values[0] <= 0:
        return dict(result, status="indefinite", reason="Mean projected Hessian is not positive definite")
    scale = 1 / np.sqrt(np.diag(mean))
    balanced = scale[:, None] * mean * scale[None, :]
    balanced_values = np.linalg.eigvalsh(balanced)
    estimates = references.mean(axis=1)
    estimates = (estimates + estimates.transpose(0, 2, 1)) / 2
    differences = [estimates[0] - mean, estimates[1] - mean, estimates[0] - estimates[1]]
    errors = [float(np.linalg.norm(scale[:, None] * delta * scale[None, :], ord=2)) for delta in differences]
    error = max(errors)
    passed = bool(balanced_values[0] > safety_factor * error)
    return dict(
        result,
        status="empirically_resolved" if passed else "unresolved",
        reason="Minimum scaled eigenvalue must exceed ten times the measured discrepancy",
        scaled_minimum_eigenvalue=float(balanced_values[0]),
        scaled_spectral_discrepancies=errors,
        margin_ratio=float(balanced_values[0] / max(error, 1e-300)),
    )
