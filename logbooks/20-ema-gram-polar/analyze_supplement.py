# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep recovered early/middle measurements separate from a validated late supplement."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from summarize_supplement import paired


def loss_index(rows, expected):
    result = {}
    for row in rows:
        if row["kind"] != "loss":
            continue
        assert row["ridge"] == 0.01
        check = paired(row["losses"], row["baseline_losses"])
        assert math.isclose(check["mean"], row["delta_mean"], abs_tol=1e-12)
        assert math.isclose(check["sem"], row["delta_sem"], abs_tol=1e-12)
        key = (row["counter"], row["alpha"], row["method"], row["step_size"])
        assert key not in result
        result[key] = row
    assert len(result) == expected
    return result


def analyze(partial, late):
    assert len(late) == 4
    partial_data = next(r for r in partial if r["kind"] == "data")
    partial_losses = loss_index(partial, 21)
    late_losses = [loss_index(rows, 9) for rows in late]
    assert all(x == late_losses[0] for x in late_losses), "Late host replicas disagree"
    data = [next(r for r in rows if r["kind"] == "data") for rows in late]
    assert sorted(r["process_index"] for r in data) == [0, 1, 2, 3]
    for r in [partial_data, *data]:
        assert (r["batch_offset"], r["batches"], r["eval_batch_size"], r["available_sequences"]) == (8, 8, 8, 141)
        assert (r["process_count"], r["device_count"]) == (4, 16)
        assert len(set(r["local_shard_sha256"])) == 8
    counterpart = next(r for r in data if r["process_index"] == partial_data["process_index"])
    assert counterpart["local_shard_sha256"] == partial_data["local_shard_sha256"]
    assert all([r["counter"] for r in rows if r["kind"] == "restore"] == [33597] for rows in late)
    assert not partial_losses.keys() & late_losses[0].keys()
    combined = partial_losses | late_losses[0]
    contrasts = []
    for counter in (3733, 18665, 33597):
        for size in (1e-5, 3e-5, 1e-4):
            quarter = combined[(counter, 0.25, "newton_schulz", size)]
            for alpha in (0.0, 0.5):
                other = combined[(counter, alpha, "newton_schulz", size)]
                assert quarter["baseline_losses"] == other["baseline_losses"]
                contrasts.append(
                    dict(
                        counter=counter,
                        step_size=size,
                        comparator_alpha=alpha,
                        provenance="validated_late" if counter == 33597 else "recovered_failed_run",
                        **paired(quarter["losses"], other["losses"]),
                    )
                )
    polar = []
    for size in (1e-5, 3e-5, 1e-4):
        ns = combined[(18665, 0.25, "newton_schulz", size)]
        svd = combined[(18665, 0.25, "svd", size)]
        assert ns["baseline_losses"] == svd["baseline_losses"]
        polar.append(dict(step_size=size, provenance="recovered_failed_run", **paired(svd["losses"], ns["losses"])))
    timers = [next(r for r in rows if r["kind"] == "matrix_direction_timings") for rows in late]
    expected_order = [(0, 0.0), (0, 0.25), (0, 0.5), (1, 0.5), (1, 0.25), (1, 0.0)]
    assert all([(r["repeat"], r["alpha"]) for r in t["rows"]] == expected_order for t in timers)
    assert all(t["counter"] == 33597 and t["state_bytes"] == timers[0]["state_bytes"] for t in timers)
    samples = []
    for index, (repeat, alpha) in enumerate(expected_order):
        seconds = [t["rows"][index]["seconds"] for t in timers]
        assert all(math.isfinite(x) and x > 0 for x in seconds)
        samples.append(dict(repeat=repeat, alpha=alpha, max_rank_seconds=max(seconds), rank_seconds=seconds))
    medians = {
        str(a): statistics.median(r["max_rank_seconds"] for r in samples if r["alpha"] == a) for a in (0.0, 0.25, 0.5)
    }
    return dict(
        status="RECOVERED_EARLY_MIDDLE_PLUS_VALIDATED_LATE",
        fresh_batches=8,
        batch_offset=8,
        norm_contrasts=contrasts,
        exact_minus_production=polar,
        timing_samples=samples,
        timing_median_max_rank_seconds=medians,
        timing_ratio_to_zero={a: v / medians["0.0"] for a, v in medians.items()},
        timing_scope=timers[0]["scope"],
        logical_matrix_state_bytes=timers[0]["state_bytes"],
        early_middle_provenance=(
            "One recovered global-loss artifact from a failed job; " "no four-host agreement or finished ONLINE claim"
        ),
        late_provenance="Four identical loss-row replicas from the validated r04 job; hosts are not extra samples",
        batch_join=(
            "Matching saved local-shard hashes for the same process index across jobs; "
            "configured global range unchanged"
        ),
        limits=[
            "No cross-seed claim",
            "Step sizes and checkpoints reuse eight batches, not independent wins",
            "SVD loss control covers only the middle checkpoint and comes from failed-run recovery",
            "Warm timings include norm-reference computation and rescaling; not full optimizer/training overhead",
            "Two timing repeats are descriptive; memory snapshots are cumulative, not alpha-specific peaks",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.validation.read_text())
    assert evidence["passed"] and evidence["canonical"]["state"] == "JOB_STATE_SUCCEEDED"
    paths = sorted(args.artifacts.glob("*/ema-gram-supplement-probe-*/rows.jsonl"))
    result = analyze(
        json.loads(args.partial.read_text()), [[json.loads(line) for line in p.read_text().splitlines()] for p in paths]
    )
    result["source_artifacts"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.partial, *paths]}
    result["late_source"] = evidence["source"]
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
