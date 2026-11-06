# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect complete asynchronous windows and compute the predeclared paired intervals."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from rigging.filesystem.buckets import filesystem_for
from scipy.stats import t


def read_result(uri):
    filesystem, path = filesystem_for(uri)
    with filesystem.open(path, "rb") as stream:
        content = stream.read()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def paired_interval(ratios):
    logs = np.log(ratios)
    mean = float(logs.mean())
    radius = float(t.ppf(0.975, len(logs) - 1) * logs.std(ddof=1) / math.sqrt(len(logs)))
    return {
        "added_percent": 100 * math.expm1(mean),
        "lower95_percent": 100 * math.expm1(mean - radius),
        "upper95_percent": 100 * math.expm1(mean + radius),
        "paired_ratios": ratios,
        "pairs": len(logs),
    }


def comparable_config(run, config):
    """Check each arm, then remove only its seed, cadence and output identities."""
    trainer = config["trainer"]
    algorithm = trainer["algorithm"]
    if (
        trainer["seed"] != run["seed"]
        or trainer["max_steps"] != 30
        or algorithm["score_centering_enabled"]
        or algorithm["score_centering_top_k"] != 128
        or algorithm["score_centering_probe"]["interval"] != run["interval"]
        or algorithm["off_policy_correction"] != "tis"
    ):
        raise ValueError(f"Resolved configuration differs from the declared arm: {run['tag']}")
    for key in ("seed", "ckpt_path", "export_path", "run_name"):
        del trainer[key]
    del algorithm["score_centering_probe"]["interval"]
    retention = config["generator"]["trajectory_retention"]
    del retention["output_path"], retention["run_id"]
    return config


def summarize(run, result, digest):
    cycles = result["cycles"]
    if [row["step"] for row in cycles] != list(range(1, 31)) or not result["optimizer_isolation"]:
        raise ValueError(f"Incomplete thirty-update window or optimizer contamination: {run['tag']}")
    lengths = np.concatenate([row["response_lengths"] for row in cycles])
    observed = [row for row in result["probes"] if row["metrics"].get("score_centering_probe/observed", 0) > 0]
    if run["interval"] and not observed:
        raise ValueError(f"No consumed nonempty probes: {run['tag']}")
    if not run["interval"] and result["probes"]:
        raise ValueError(f"Disabled-probe control contains probe records: {run['tag']}")
    selected_lengths = [length for row in observed for length in row["response_lengths"]]
    counts = [count for row in observed for count in row["sampled_positions_per_response"]]
    if counts and max(counts) > 32:
        raise ValueError("Probe coverage exceeds the declared per-response limit")
    gradients = [row["metrics"]["policy/raw_grad_norm"] for row in cycles]
    learning_rates = [row["metrics"]["policy/policy_lr"] for row in cycles]
    published_steps = {row["step"] for row in result["publications"]}
    if not all(math.isfinite(value) and value >= 0 for value in gradients) or max(gradients) == 0:
        raise ValueError(f"No finite nonzero optimizer updates: {run['tag']}")
    if not all(math.isfinite(value) and value > 0 for value in learning_rates):
        raise ValueError(f"Nonpositive learning rate: {run['tag']}")
    if not set(range(1, 31)).issubset(published_steps):
        raise ValueError(f"Missing completed weight publications: {run['tag']}")
    loss_tokens = sum(row["trained_loss_tokens"] for row in cycles)
    response_tokens = sum(row["response_tokens"] for row in cycles)
    elapsed = cycles[-1]["complete_cycle_elapsed_seconds"]
    token_age = (
        sum(row["metrics"]["async/staleness_tokens_mean"] * row["trained_loss_tokens"] for row in cycles) / loss_tokens
    )
    fully_stamped = all(row["metrics"]["async/staleness_tokens_stamped"] == 1 for row in cycles)
    summary = {
        **run,
        "completed_sha256": digest,
        "complete_cycle_seconds": elapsed,
        "mean_cycle_seconds": elapsed / len(cycles),
        "warmup_three_cycles_seconds": cycles[2]["complete_cycle_elapsed_seconds"],
        "warmed_27_cycles_seconds": elapsed - cycles[2]["complete_cycle_elapsed_seconds"],
        "loss_tokens": loss_tokens,
        "response_tokens": response_tokens,
        "seconds_per_million_loss_tokens": elapsed / loss_tokens * 1e6,
        "seconds_per_million_response_tokens": elapsed / response_tokens * 1e6,
        "response_length_p50_p95_p99": np.quantile(lengths, [0.5, 0.95, 0.99]).tolist(),
        "response_length_max": int(lengths.max()),
        "responses_ge2048": int((lengths >= 2048).sum()),
        "mean_token_staleness": token_age if fully_stamped else None,
        "max_token_staleness": (
            max(row["metrics"]["async/staleness_tokens_max"] for row in cycles) if fully_stamped else None
        ),
        "token_stamp_coverage": (
            sum(row["metrics"]["async/staleness_tokens_stamped"] * row["trained_loss_tokens"] for row in cycles)
            / loss_tokens
        ),
        "raw_gradient_norm_min_max": [min(gradients), max(gradients)],
        "learning_rate_min_max": [min(learning_rates), max(learning_rates)],
        "published_policy_steps": sorted(published_steps),
        "consumed_probe_events": len(observed),
        "consumed_probe_responses": len(selected_lengths),
        "consumed_probe_positions": sum(len(row["positions"]) for row in observed),
        "long_response_probe_positions": sum(row["long_response_positions"] for row in observed),
        "probe_response_lengths": selected_lengths,
        "probe_source_policy_steps": [sorted(set(row["source_policy_steps"])) for row in observed],
        "probe_training_steps": [row["training_step"] for row in observed],
    }
    return summary


def analyze(manifest, output):
    output.mkdir(parents=True, exist_ok=True)
    admitted = [run for run in manifest["runs"] if run["kind"] == "grug" and run["admitted"]]
    seeds = (1701, 2903, 4517)
    keys = {(run["seed"], run["interval"]) for run in admitted}
    if len(keys) != len(admitted):
        raise ValueError("Repeated admitted seed/interval window")
    if any((seed, interval) not in keys for seed in seeds for interval in (0, 10)):
        raise ValueError("Need one admitted off and interval-ten window for each declared seed")
    summaries, rows, common_config = [], [], None
    for run in admitted:
        result, digest = read_result(run["completed_uri"])
        source, _ = read_result(run["runtime_source_uri"])
        if source != manifest["runtime_source_hashes"]:
            raise ValueError(f"Measured runtime source differs from the pinned cohort: {run['tag']}")
        summary = summarize(run, result, digest)
        config_uri = run["completed_uri"].rsplit("/", 1)[0] + "/config.json"
        config, config_digest = read_result(config_uri)
        config = comparable_config(run, config)
        if common_config is None:
            common_config = config
        elif config != common_config:
            raise ValueError(f"Resolved settings differ between cohort windows: {run['tag']}")
        summary["config_sha256"] = config_digest
        summary["common_config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        summaries.append(summary)
        previous = 0
        for cycle in result["cycles"]:
            elapsed = cycle["complete_cycle_elapsed_seconds"]
            rows.append(
                {
                    "tag": run["tag"],
                    "seed": run["seed"],
                    "interval": run["interval"],
                    "step": cycle["step"],
                    "elapsed_seconds": elapsed,
                    "cycle_seconds": elapsed - previous,
                    "loss_tokens": cycle["trained_loss_tokens"],
                    "response_tokens": cycle["response_tokens"],
                    "token_staleness": (
                        cycle["metrics"]["async/staleness_tokens_mean"]
                        if cycle["metrics"]["async/staleness_tokens_stamped"] == 1
                        else None
                    ),
                    "token_stamp_coverage": cycle["metrics"]["async/staleness_tokens_stamped"],
                }
            )
            previous = elapsed
    by_arm = {(run["seed"], run["interval"]): run for run in summaries}
    comparisons = {}
    for interval in sorted({run["interval"] for run in summaries} - {0}):
        if any((seed, arm) not in by_arm for seed in seeds for arm in (0, interval)):
            raise ValueError(f"Incomplete paired comparison for interval {interval}")
        comparisons[str(interval)] = {
            metric: paired_interval([by_arm[seed, interval][metric] / by_arm[seed, 0][metric] for seed in seeds])
            for metric in (
                "complete_cycle_seconds",
                "warmed_27_cycles_seconds",
                "seconds_per_million_loss_tokens",
                "seconds_per_million_response_tokens",
            )
        }
    result = {"runs": summaries, "comparisons": comparisons, "method": manifest["protocol_uri"]}
    (output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (output / "per-cycle.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(json.loads(args.manifest.read_text()), args.output)
