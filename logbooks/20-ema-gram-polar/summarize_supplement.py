# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize generation-verified supplemental artifacts without treating hosts as samples."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def paired(left, right):
    differences = [a - b for a, b in zip(left, right, strict=True)]
    if len(differences) != 8 or not all(math.isfinite(x) for x in differences):
        raise ValueError("Expected eight finite paired batch differences")
    return dict(
        batch_differences=differences,
        mean=statistics.mean(differences),
        std=statistics.stdev(differences),
        sem=statistics.stdev(differences) / math.sqrt(len(differences)),
    )


def summarize(paths):
    if len(paths) != 4:
        raise ValueError("Expected exactly four replica artifacts")
    ranks = []
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        data = [r for r in rows if r["kind"] == "data"]
        assert len(data) == 1 and data[0]["batch_offset"] == 8 and data[0]["batches"] == 8
        assert data[0]["process_count"] == 4 and data[0]["device_count"] == 16
        assert len(data[0]["local_shard_sha256"]) == 8
        restored = [r["counter"] for r in rows if r["kind"] == "restore"]
        assert restored == [3733, 18665, 33597]
        losses = [r for r in rows if r["kind"] == "loss"]
        assert len(losses) == 30
        keyed = {}
        for row in losses:
            assert row["ridge"] == 0.01
            check = paired(row["losses"], row["baseline_losses"])
            assert math.isclose(check["mean"], row["delta_mean"], abs_tol=1e-12)
            assert math.isclose(check["sem"], row["delta_sem"], abs_tol=1e-12)
            key = (row["counter"], row["alpha"], row["method"], row["step_size"])
            assert key not in keyed
            keyed[key] = row
        ranks.append(dict(data=data[0], rows=rows, keyed=keyed))
    assert sorted(r["data"]["process_index"] for r in ranks) == list(range(4))
    reference = ranks[0]["keyed"]
    for rank in ranks[1:]:
        assert rank["keyed"] == reference, "Host replicas disagree; investigate before scientific aggregation"
    contrasts = []
    for counter in (3733, 18665, 33597):
        for size in (1e-5, 3e-5, 1e-4):
            quarter = reference[(counter, 0.25, "newton_schulz", size)]
            for alpha in (0.0, 0.5):
                control = reference[(counter, alpha, "newton_schulz", size)]
                assert control["baseline_losses"] == quarter["baseline_losses"]
                contrasts.append(
                    dict(
                        counter=counter,
                        step_size=size,
                        comparator_alpha=alpha,
                        **paired(quarter["losses"], control["losses"]),
                    )
                )
    polar = []
    for size in (1e-5, 3e-5, 1e-4):
        production = reference[(18665, 0.25, "newton_schulz", size)]
        exact = reference[(18665, 0.25, "svd", size)]
        assert production["baseline_losses"] == exact["baseline_losses"]
        polar.append(dict(counter=18665, step_size=size, **paired(exact["losses"], production["losses"])))
    timings = [next(r for r in rank["rows"] if r["kind"] == "matrix_direction_timings") for rank in ranks]
    samples = []
    for index, row in enumerate(timings[0]["rows"]):
        peers = [t["rows"][index] for t in timings]
        assert all((p["alpha"], p["repeat"]) == (row["alpha"], row["repeat"]) for p in peers)
        assert all(math.isfinite(p["seconds"]) and p["seconds"] > 0 for p in peers)
        samples.append(dict(alpha=row["alpha"], repeat=row["repeat"], max_rank_seconds=max(p["seconds"] for p in peers)))
    medians = {
        str(a): statistics.median(r["max_rank_seconds"] for r in samples if r["alpha"] == a) for a in (0.0, 0.25, 0.5)
    }
    return dict(
        source_artifacts={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        replica_agreement="All four hosts' thirty loss rows agree exactly; aggregate one copy",
        fresh_batches=8,
        batch_offset=8,
        norm_contrasts=contrasts,
        exact_minus_production=polar,
        timing_samples=samples,
        timing_median_max_rank_seconds=medians,
        timing_ratio_to_zero={a: value / medians["0.0"] for a, value in medians.items()},
        timing_scope=(
            "Saved matrix directions with the probe's default eigensolver layout; "
            "excludes gradient/Gram updates and training matrix-parallel layout"
        ),
        logical_matrix_state_bytes=timings[0]["state_bytes"],
        memory=[
            dict(process_index=r["data"]["process_index"], records=[x for x in r["rows"] if x["kind"] == "memory"])
            for r in ranks
        ],
        limits=[
            "Batch-conditional local perturbations, not cross-seed or full-training effects",
            "Steps and checkpoints reuse the same eight batches; do not count independent wins",
            "Exact SVD loss control at one prespecified middle checkpoint only",
            "Cumulative memory includes restore, probes, compilation and SVD; not alpha-specific peak training memory",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.artifacts.glob("*/ema-gram-supplement-probe-*/rows.jsonl"))
    result = summarize(paths)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
