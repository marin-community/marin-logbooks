# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare held-out replay loss between Snowball checkpoints, split by junk-filter verdict.

Loss is token-weighted within a group: summed loss over summed scored tokens. A difference between two
checkpoints uses the same documents for both and is bootstrapped by resampling documents within each
verdict group (10,000 resamples, seed 0). "Flagged minus kept" is the difference of the two groups'
differences, the check on whether a checkpoint lost more on flagged documents than on kept ones.

usage: paired_replay_loss.py <run id>... (the first is the reference for every comparison)
"""

import json
import sys

import numpy as np
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

ROOT = "s3://marin-us-east-02a/marin/eval/replay-loss-snowball"
VERSION = "2026.11.16"
RESAMPLES = 10_000


def load(run: str) -> dict[tuple[str, int], dict]:
    path = StoragePath.parse(f"{ROOT}/{run}/{VERSION}/replay_document_losses.jsonl")
    return {(row["cell"], row["index"]): row for row in map(json.loads, path.read_text().splitlines())}


def group_loss(rows: list[dict]) -> float:
    return sum(r["loss_sum"] for r in rows) / sum(r["scored_tokens"] for r in rows)


def paired(a: list[dict], b: list[dict], rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Token-weighted b minus a, and its bootstrap draws, over documents in the same order."""
    delta = np.array([rb["loss_sum"] - ra["loss_sum"] for ra, rb in zip(a, b, strict=True)])
    tokens = np.array([ra["scored_tokens"] for ra in a], dtype=float)
    draws = rng.integers(0, len(a), size=(RESAMPLES, len(a)))
    return delta.sum() / tokens.sum(), delta[draws].sum(axis=1) / tokens[draws].sum(axis=1)


def interval(draws: np.ndarray) -> str:
    low, high = np.quantile(draws, [0.025, 0.975])
    return f"({low:+.4f} to {high:+.4f})"


def main() -> None:
    configure_coreweave_s3()
    runs = sys.argv[1:]
    data = {run: load(run) for run in runs}
    keys = sorted(set.intersection(*(set(d) for d in data.values())))
    groups = {
        "kept": [k for k in keys if data[runs[0]][k]["category"] is None],
        "flagged": [k for k in keys if data[runs[0]][k]["category"] is not None],
    }
    print(f"documents={len(keys)} kept={len(groups['kept'])} flagged={len(groups['flagged'])}")
    for run in runs:
        losses = {name: group_loss([data[run][k] for k in ks]) for name, ks in groups.items()}
        print(f"{run}: kept={losses['kept']:.4f} flagged={losses['flagged']:.4f}")
    reference = runs[0]
    for run in runs[1:]:
        rng = np.random.default_rng(0)
        results = {}
        for name, ks in groups.items():
            results[name] = paired([data[reference][k] for k in ks], [data[run][k] for k in ks], rng)
        gap = results["flagged"][0] - results["kept"][0]
        gap_draws = results["flagged"][1] - results["kept"][1]
        print(
            f"{run} minus {reference}: kept {results['kept'][0]:+.4f} {interval(results['kept'][1])}, "
            f"flagged {results['flagged'][0]:+.4f} {interval(results['flagged'][1])}, "
            f"flagged minus kept {gap:+.4f} {interval(gap_draws)}"
        )


if __name__ == "__main__":
    main()
