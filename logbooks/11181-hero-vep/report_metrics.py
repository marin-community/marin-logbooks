# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///

"""Report the pinned MarinDNA metrics on complete missense matched groups.

Put the pinned MarinDNA evals_v2/src directory on PYTHONPATH before execution.
"""

import argparse
import hashlib
import json
from pathlib import Path

import marin_dna_evals.metrics as native_metrics
import numpy as np
import pandas as pd

METRICS_SHA256 = "bee720324d61ea8b88a4644b6bd036569e9bab6f8e931d6335d165fb2e2b2395"
METHODS = ["null", "healthy", "zero-shot", "two-shot", "zero-shot/mapping-average", "two-shot/mapping-average"]


def report(frame: pd.DataFrame, n_bootstrap: int, seed: int) -> dict:
    assert hashlib.sha256(Path(native_metrics.__file__).read_bytes()).hexdigest() == METRICS_SHA256
    assert n_bootstrap >= 2
    assert set(frame["subset"]) == {"missense_variant"}
    assert frame["id"].is_unique
    assert np.isfinite(frame[METHODS].to_numpy()).all()
    grouped = frame.groupby("match_group")
    assert (grouped.size() == 10).all() and (grouped["label"].sum() == 1).all()
    groups = int(frame["match_group"].nunique())
    result = {
        "n_groups": groups,
        "exploratory": groups < 30,
        "bootstrap_replicates": n_bootstrap,
        "bootstrap_seed": seed,
        "metrics_source_sha256": METRICS_SHA256,
        "auprc": {},
        "paired_deltas": {},
        "within_group_pairwise_accuracy": {},
    }
    for method in METHODS:
        result["auprc"][method] = native_metrics.auprc_with_bootstrap_se(
            frame["label"],
            frame[method],
            frame["match_group"],
            n_bootstrap=n_bootstrap,
            rng=seed,
        )
        wins = []
        for _, group in grouped:
            positive = group.loc[group["label"].astype(bool), method].iloc[0]
            controls = group.loc[~group["label"].astype(bool), method]
            wins.extend((positive > controls).astype(float) + 0.5 * (positive == controls))
        result["within_group_pairwise_accuracy"][method] = float(np.mean(wins))
    for treatment, control in (
        ("healthy", "null"),
        ("zero-shot", "null"),
        ("two-shot", "zero-shot"),
        ("two-shot/mapping-average", "zero-shot/mapping-average"),
    ):
        result["paired_deltas"][f"{treatment} minus {control}"] = native_metrics.paired_metric_delta_bootstrap(
            frame["label"],
            frame[treatment],
            frame[control],
            frame["match_group"],
            n_bootstrap=n_bootstrap,
            rng=seed,
        )
    if groups >= 30:
        result["standard_subset_metrics"] = native_metrics.compute_auprc_metrics(
            frame[["label", "subset", "match_group"]],
            frame[METHODS],
            n_bootstrap=n_bootstrap,
            rng=seed,
        ).to_dict(orient="records")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    frame = pd.DataFrame(json.loads(args.variant_scores.read_text()))
    result = report(frame, args.bootstrap_replicates, args.seed)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
