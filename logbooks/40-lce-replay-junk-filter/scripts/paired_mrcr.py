"""Paired comparison of two Snowball MRCR runs over the same sources.

Needle recovery is token-weighted: the summed gain from the needles when distractors are present
(distractor-only loss minus full-context loss) over the summed gain without them (query-only minus
needle-only). The difference between two runs is bootstrapped by resampling sources, keeping each
source's rows for both runs together, as marin#9519 did.

usage: paired_mrcr.py <baseline run id> <treatment run id> [resamples]
"""

import json
import sys

import numpy as np
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

ROOT = "s3://marin-us-east-08a/marin/eval/mrcr-snowball"
SEED = 0


def load(run: str) -> dict[str, dict]:
    path = StoragePath.parse(f"{ROOT}/{run}/2026.11.13/mrcr_example_losses.jsonl")
    return {row["source_id"]: row for row in map(json.loads, path.read_text().splitlines())}


def gains(rows: list[dict]) -> np.ndarray:
    """Columns: distracted gain, isolated gain, context gain (query-only minus full), scored tokens."""
    return np.array(
        [
            (
                r["distractor_only_loss_sum"] - r["full_context_loss_sum"],
                r["query_only_loss_sum"] - r["needle_only_loss_sum"],
                r["query_only_loss_sum"] - r["full_context_loss_sum"],
                r["scored_tokens"],
            )
            for r in rows
        ]
    )


def stats(g: np.ndarray) -> tuple[float, float, float]:
    totals = g.sum(axis=0)
    return totals[0] / totals[1], totals[0] / totals[3], totals[2] / totals[3]


def main() -> None:
    configure_coreweave_s3()
    baseline_run, treatment_run = sys.argv[1:3]
    resamples = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000
    baseline, treatment = load(baseline_run), load(treatment_run)
    sources = sorted(baseline)
    if sources != sorted(treatment):
        raise ValueError("the two runs scored different sources")
    a = gains([baseline[s] for s in sources])
    b = gains([treatment[s] for s in sources])
    names = ("needle recovery", "distracted gain, nats/token", "context gain, nats/token")
    point_a, point_b = stats(a), stats(b)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(sources), size=(resamples, len(sources)))
    deltas = np.array([np.subtract(stats(b[idx]), stats(a[idx])) for idx in draws])
    print(f"{treatment_run} minus {baseline_run}, {len(sources)} sources, {resamples} paired resamples")
    for i, name in enumerate(names):
        low, high = np.percentile(deltas[:, i], [2.5, 97.5])
        share = float((deltas[:, i] > 0).mean())
        print(
            f"  {name:28s} {point_a[i]:.4f} -> {point_b[i]:.4f}  delta {point_b[i] - point_a[i]:+.4f} "
            f"[{low:+.4f}, {high:+.4f}]  resamples above zero {share:.1%}"
        )
    better = sum(
        (rb[0] / rb[1] if rb[1] else 0) > (ra[0] / ra[1] if ra[1] else 0) for ra, rb in zip(a, b, strict=True)
    )
    print(f"  sources where the treatment's own recovery is higher: {better} of {len(sources)}")


if __name__ == "__main__":
    main()
