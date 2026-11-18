# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paired comparison of generative MRCR-65k accuracy between the three step-238 checkpoints.

Reads the per-item samples of each evaluation, keeps items graded in all three, and bootstraps the
mean difference over items (10,000 resamples, seed 0). Items without a grade (empty output) are dropped.

usage: paired_mrcr_generative.py
"""

import json

import numpy as np
import pandas as pd
from finestore.reader import ReadView
from rigging.filesystem.s3_compat import configure_coreweave_s3

ROOT = "s3://marin-us-east-02a/marin/evals"
RUNS = {
    "control seed 0": (
        "20261117-233607-lce40-arm-control-step238-mrcr-483a0ac6be9b4095a0b45d42f89582ba"
        "-mrcr-65k-4a67097a26dd525e896f91c0e755cf9e"
    ),
    "control seed 1": (
        "20261117-233613-lce40-arm-control-seed1-step238-mrcr-3729dbe0b1f84cf688c265a37be42b8a"
        "-mrcr-65k-2a03b178da1b5d1b977ba8f7600b1c69"
    ),
    "filtered": (
        "20261117-233619-lce40-arm-filtered-step238-mrcr-014c3a2ff03e469fb25f94c2f958af8f"
        "-mrcr-65k-24c95d9b8163576795f6f1dd36e33af9"
    ),
}
RESAMPLES = 10_000


def item_scores(run: str) -> pd.DataFrame:
    samples = ReadView(f"{ROOT}/{run}/results").scan("samples")
    samples = samples.to_pandas() if hasattr(samples, "to_pandas") else samples
    rows = []
    for _, sample in samples.iterrows():
        metrics = dict(sample["metrics"])
        if "accuracy" not in metrics:
            continue
        doc = sample["doc"] if isinstance(sample["doc"], dict) else json.loads(sample["doc"])
        rows.append(
            {
                "doc_id": sample["doc_id"],
                "accuracy": metrics["accuracy"],
                "prefix": metrics["prefix_hit_rate"],
                "cell": doc["groups"]["cell"],
            }
        )
    return pd.DataFrame(rows).set_index("doc_id")


def interval(values: np.ndarray, rng: np.random.Generator) -> str:
    draws = values[rng.integers(0, len(values), (RESAMPLES, len(values)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return f"{values.mean():+.4f} ({low:+.4f} to {high:+.4f})"


def main() -> None:
    configure_coreweave_s3()
    scores = {name: item_scores(run) for name, run in RUNS.items()}
    table = pd.DataFrame({name: frame["accuracy"] for name, frame in scores.items()}).dropna()
    cells = scores["filtered"]["cell"].loc[table.index]
    print(f"items graded in all three: {len(table)}")
    for name, frame in scores.items():
        print(
            f"{name}: accuracy {table[name].mean():.3f}, prefix hit {frame['prefix'].mean():.3f}, "
            f"near-exact {(table[name] >= 0.9).sum()}"
        )
    control_mean = (table["control seed 0"] + table["control seed 1"]) / 2
    rng = np.random.default_rng(0)
    comparisons = {
        "filtered minus seed 0": table["filtered"] - table["control seed 0"],
        "filtered minus seed 1": table["filtered"] - table["control seed 1"],
        "filtered minus control mean": table["filtered"] - control_mean,
        "seed 1 minus seed 0": table["control seed 1"] - table["control seed 0"],
    }
    for label, difference in comparisons.items():
        print(f"{label}: {interval(difference.to_numpy(), rng)}")
    needles = cells.str.extract(r"_(\d)needle")[0].astype(int).rename("needles")
    print(table.groupby(needles).mean().round(3))


if __name__ == "__main__":
    main()
