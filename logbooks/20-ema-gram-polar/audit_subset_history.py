# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit stored metric rows only; never declare scientific completion from W&B alone."""
import math

TRAIN_KEYS = ("train/loss", "throughput/duration", "throughput/mfu", "optim/learning_rate", "optim/adam_lr")
FINAL_KEY = "eval/paloma/c4_en/loss"


def audit(history, expected_steps=7630):
    metrics = {key: {} for key in TRAIN_KEYS}
    evaluations = {}
    for row in history:
        step = row.get("global_step")
        if step is None:
            continue
        if isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step) or int(step) != step:
            raise ValueError("Invalid native step label")
        step = int(step)
        for key in (*TRAIN_KEYS, FINAL_KEY):
            value = row.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Nonfinite/invalid {key} at {step}")
            target = evaluations if key == FINAL_KEY else metrics[key]
            if step in target:
                raise ValueError(f"Duplicate {key} at {step}")
            target[step] = value
    required = set(range(expected_steps))
    for key, rows in metrics.items():
        if set(rows) != required:
            raise ValueError(f"Incomplete or extraneous step labels for {key}")
    if any(value <= 0 for value in metrics["throughput/duration"].values()):
        raise ValueError("Nonpositive update duration")
    if expected_steps - 1 not in evaluations:
        raise ValueError("Missing final same-budget C4 evaluation")
    return dict(
        metric_history_complete=True,
        scientifically_complete=False,
        expected_steps=expected_steps,
        native_final_label=expected_steps - 1,
        final_c4_loss=evaluations[expected_steps - 1],
        train_metric_rows={key: len(rows) for key, rows in metrics.items()},
        remaining_gates=[
            "canonical all-task/all-attempt outcome",
            "exact source/data/config and run-attempt joins",
            "native final counter and checkpoint metadata",
            "ONLINE finished state",
            "verified full-state restoration for any resumed intervals",
        ],
    )
