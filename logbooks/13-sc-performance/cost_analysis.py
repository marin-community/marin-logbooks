# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize interleaved cost blocks; keep raw records as the audit trail."""

import argparse
import json
from pathlib import Path

import numpy as np
from rigging.filesystem.buckets import filesystem_for


def read_json(root, name):
    location = root.rstrip("/") + "/" + name
    if "://" not in location:
        return json.loads(Path(location).read_text())
    filesystem, path = filesystem_for(location)
    with filesystem.open(path, "r") as source:
        return json.load(source)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
    }


def relative_cost(rows, baseline, name="seconds"):
    blocks = {row["repeat"]: row[name] for row in rows}
    reference = {row["repeat"]: row[name] for row in baseline}
    assert len(blocks) == len(rows) and len(reference) == len(baseline)
    assert set(blocks) == set(reference)
    pairs = np.asarray([[blocks[key], reference[key]] for key in sorted(blocks)], dtype=np.float64)
    assert np.all(pairs > 0)
    indices = np.random.default_rng(41).integers(len(pairs), size=(10000, len(pairs)))
    means = pairs[indices].mean(axis=1)
    bootstrap = 100 * (means[:, 0] / means[:, 1] - 1)
    return {
        "percent_ratio_of_means": float(100 * (pairs[:, 0].mean() / pairs[:, 1].mean() - 1)),
        "paired_block_bootstrap_95_percentile": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "paired_block_percent_changes": (100 * (pairs[:, 0] / pairs[:, 1] - 1)).tolist(),
    }


def training_summary(records):
    steady = [row for row in records if row["repeat"] >= 0]
    baseline = [row for row in steady if row["arm"] == "tis"]
    results = {}
    for arm in dict.fromkeys(row["arm"] for row in records):
        rows = [row for row in steady if row["arm"] == arm]
        rank_counts = {len(row["memory"]["ranks"]) for row in rows}
        assert len(rank_counts) == 1
        metrics = sorted({key for row in rows for key in row["metrics"]})
        selected = [
            key
            for key in metrics
            if key.startswith(("score_centering_capture/", "timing/score_centering", "async/performance/consumed_"))
            or key
            in ("generate/avg_num_tokens", "generate/max_num_tokens", "timing/run_training", "timing/policy_train")
        ]
        result = {
            "seconds": distribution([row["seconds"] for row in rows]),
            "relative_to_tis": relative_cost(rows, baseline),
            "learner_ranks": next(iter(rank_counts)),
            "physical_devices": sorted({sum(len(n["used_bytes"]) for n in row["memory"]["devices"]) for row in rows}),
            "torch_peak_allocated_gib": distribution(
                [max(rank["allocated"] for rank in row["memory"]["ranks"]) / 2**30 for row in rows]
            ),
            "torch_peak_reserved_gib": distribution(
                [max(rank["reserved"] for rank in row["memory"]["ranks"]) / 2**30 for row in rows]
            ),
            "physical_peak_used_gib": distribution(
                [max(value for node in row["memory"]["devices"] for value in node["used_bytes"]) / 2**30 for row in rows]
            ),
            "probe_consumption": {
                key.rsplit("/", 1)[-1]: sum(row["metrics"].get(key, 0) for row in rows)
                for key in (
                    "score_centering_probe/observed",
                    "score_centering_probe/sampled_responses",
                    "score_centering_probe/sampled_tokens",
                )
            },
            "metrics_mean": {key: float(np.mean([row["metrics"].get(key, 0) for row in rows])) for key in selected},
        }
        if "timings" in rows[0]:
            result["timings_mean_seconds"] = {
                key: float(np.mean([row["timings"].get(key, 0) for row in rows]))
                for key in sorted({key for row in rows for key in row["timings"]})
            }
            assert len({row["batch_hash"] for row in steady}) == 1
            assert len({row["loss_tokens"] for row in steady}) == 1
            result["batch_hash"] = rows[0]["batch_hash"]
            result["loss_tokens"] = rows[0]["loss_tokens"]
        results[arm] = result
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {
        "source_root": args.root,
        "limits": [
            "Warmup block -1 excluded; uncertainty resamples whole paired repeat blocks, with seed 41.",
            "Small repeat counts give weak uncertainty estimates; percentile intervals do not establish significance.",
            "Live cycles contain stochastic responses; matched trainer comparisons hold tokens and routes fixed.",
            "Serialized fixed-weight costs do not measure asynchronous throughput or learning benefit.",
            "Memory summarizes the maximum across retained ranks/devices; physical peaks are sampled every 50 ms.",
            "Capture durations sum request lifetimes and are not exclusive GPU time or serialized cycle stages.",
            "Payload byte counts are logical arrays, not measured wire traffic.",
        ],
        "engine_runtime": read_json(args.root, "engine-runtime.json"),
        "live": training_summary(read_json(args.root, "live-cycles.json")),
        "matched": training_summary(read_json(args.root, "matched-training.json")),
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {kind: {arm: row["relative_to_tis"] for arm, row in report[kind].items()} for kind in ("live", "matched")}
        )
    )


if __name__ == "__main__":
    main()
