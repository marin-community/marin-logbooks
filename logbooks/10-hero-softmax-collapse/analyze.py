# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the archived metrics and write checkpoint comparison rows."""

import argparse
import csv
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("summary.csv"))
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    rows = []
    seen = set()
    for entry in result["results"]:
        checkpoint = entry["checkpoint"]
        step = checkpoint["step"]
        assert step not in seen, f"Duplicate checkpoint step {step}"
        seen.add(step)
        metrics = entry["metrics"]
        for key, value in metrics.items():
            values = value if isinstance(value, list) else [value]
            assert all(math.isfinite(v) for v in values), f"Nonfinite metric {key} at {step}"
        sampled = metrics["valid_count"]
        production = metrics["production_valid_count"]
        assert sampled > 0 and production > 0
        assert sum(metrics["margin_histogram"]) == sampled
        expected = next(c for c in result["request"]["checkpoints"] if c["step"] == step)
        assert checkpoint == expected, f"Checkpoint metadata mismatch at {step}"
        rows.append(
            {
                "step": step,
                "production_valid_tokens": production,
                "production_zero_ce_tokens": metrics["production_zero_ce_count"],
                "production_zero_ce_percent": 100 * metrics["production_zero_ce_count"] / production,
                "sampled_tokens": sampled,
                "sampled_zero_ce_positive_tail": metrics["sampled_production_zero_ce_with_positive_tail_count"],
                "sampled_zero_ce_positive_tail_percent": (
                    100 * metrics["sampled_production_zero_ce_with_positive_tail_count"] / sampled
                ),
                "sampled_margin_above_fp32_cliff": metrics["margin_above_fp32_cliff_count"],
                "production_mean_ce": metrics["production_ce_sum"] / production,
                "sampled_mean_stable_ce": metrics["stable_ce_sum"] / sampled,
                "mean_dense_ce_absolute_error": metrics["dense_fp32_ce_absolute_error_sum"] / sampled,
                "mean_reconstructed_production_ce_gradient_imbalance": (
                    metrics["sampled_production_gradient_imbalance_sum"] / sampled
                ),
                "mean_max_logit": metrics["max_logit_sum"] / sampled,
                "mean_min_logit": metrics["min_logit_sum"] / sampled,
                "mean_target_logit": metrics["target_logit_sum"] / sampled,
                "mean_margin": metrics["margin_sum"] / sampled,
                "production_nonfinite_tokens": metrics["production_nonfinite_count"],
            }
        )
    assert seen == {c["step"] for c in result["request"]["checkpoints"]}
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
