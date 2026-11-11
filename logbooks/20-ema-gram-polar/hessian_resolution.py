# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report unshifted curvature spectra and conditional batch stability."""

import argparse
import json
from pathlib import Path

import numpy as np


def diagnostics(path, repetitions=1000, seed=20261110):
    with np.load(path, allow_pickle=False) as data:
        batches = data["curvature_batches"]
    asymmetries = np.linalg.norm(batches - batches.transpose(0, 2, 1), axis=(1, 2)) / np.linalg.norm(
        batches, axis=(1, 2)
    )
    batches = (batches + batches.transpose(0, 2, 1)) / 2
    mean = batches.mean(axis=0)
    values = np.linalg.eigvalsh(mean)
    rng = np.random.default_rng(seed)
    minima = [
        np.linalg.eigvalsh(batches[rng.integers(len(batches), size=len(batches))].mean(axis=0))[0]
        for _ in range(repetitions)
    ]
    result = dict(
        spectra_target="Symmetric part for diagnostics only; does not override the supplied analysis gates",
        per_group_relative_asymmetry=asymmetries.tolist(),
        passes_input_symmetry_gate=bool(np.max(asymmetries) <= 1e-5),
        mean_eigenvalues=values.tolist(),
        per_group_minimum_eigenvalues=np.linalg.eigvalsh(batches)[:, 0].tolist(),
        bootstrap_minimum_eigenvalue_percentile95=np.quantile(minima, [0.025, 0.975]).tolist(),
        bootstrap_nonpositive_count=int(np.count_nonzero(np.asarray(minima) <= 0)),
        bootstrap_repetitions=repetitions,
        bootstrap_seed=seed,
        relative_positivity_gate={str(t): bool(values[0] > t * values[-1]) for t in [1e-12, 1e-10, 1e-8, 1e-7, 1e-6]},
        warning=(
            "Bootstrap conditions on this saved EMA and chosen subspace. "
            "Relative eigenvalue thresholds are sensitivity checks, not calibrated error bounds."
        ),
    )
    if np.all(np.diag(mean) > 0):
        scale = 1 / np.sqrt(np.diag(mean))
        result["diagonally_scaled_mean_eigenvalues"] = np.linalg.eigvalsh(
            scale[:, None] * mean * scale[None, :]
        ).tolist()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    results = {name: diagnostics(args.directory / (name + ".npz")) for name in ("attention", "mlp")}
    (args.directory / "resolution-diagnostics.json").write_text(json.dumps(results, indent=2) + "\n")
