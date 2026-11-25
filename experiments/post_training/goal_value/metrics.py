# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline value metrics with the task as the independent sampling unit."""

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from experiments.datakit.cluster.quality.fast_transformer.metrics import auc

PREDICTORS = ("value", "task_only", "global_rate", "family_rate")


def losses(outcome: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return (probability - outcome) ** 2, -(outcome * np.log(probability) + (1 - outcome) * np.log1p(-probability))


def scores(outcome: np.ndarray, probability: np.ndarray) -> dict:
    brier, nll = losses(outcome, probability)
    mixed = 0 < outcome.sum() < len(outcome)
    return {
        "brier": float(brier.mean()),
        "log_loss": float(nll.mean()),
        "roc_auc": auc(outcome.tolist(), probability.tolist()) if mixed else None,
        "average_precision": average_precision(outcome, probability) if mixed else None,
        "mean_prediction": float(probability.mean()),
    }


def average_precision(outcome: np.ndarray, probability: np.ndarray) -> float:
    """Compute the step integral of precision against recall using distinct score thresholds."""
    order = np.argsort(-probability)
    ends = np.r_[np.flatnonzero(np.diff(probability[order])), len(order) - 1]
    true_positives = np.cumsum(outcome[order])[ends]
    precision = true_positives / (ends + 1)
    return float(np.dot(np.diff(true_positives, prepend=0), precision) / true_positives[-1])


def calibration(outcome: np.ndarray, probability: np.ndarray) -> list[dict]:
    bins = np.minimum((probability * 10).astype(int), 9)
    return [
        {
            "bin": i,
            "count": int(mask.sum()),
            "mean_prediction": float(probability[mask].mean()),
            "observed_rate": float(outcome[mask].mean()),
        }
        for i in range(10)
        if (mask := bins == i).any()
    ]


def bootstrap_means(counts: np.ndarray, sums: np.ndarray) -> list:
    """Paired interval over task clusters while retaining the turn-weighted estimand."""
    rng = np.random.default_rng(17)
    selected = rng.integers(0, len(counts), size=(2000, len(counts)))
    estimates = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return np.quantile(estimates, [0.025, 0.975], axis=0).tolist()


def paired_difference(rows: list[dict], other: str) -> dict:
    y = np.array([r["outcome"] for r in rows])
    task_ids, task_index = np.unique([r["task_id"] for r in rows], return_inverse=True)
    counts = np.bincount(task_index)
    first = losses(y, np.array([r["value"] for r in rows]))
    second = losses(y, np.array([r[other] for r in rows]))
    differences = [a - b for a, b in zip(first, second, strict=True)]
    sums = np.column_stack([np.bincount(task_index, weights=delta) for delta in differences])
    interval = bootstrap_means(counts[:, None], sums)
    return {
        "tasks": len(task_ids),
        "brier": {"delta": float(differences[0].mean()), "ci95": [interval[0][0], interval[1][0]]},
        "log_loss": {"delta": float(differences[1].mean()), "ci95": [interval[0][1], interval[1][1]]},
    }


def matched_turns(rows: list[dict]) -> dict:
    """Compare successful and failed attempts at matched ordinals within a checkpoint."""
    by_task: dict = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["turn"] > 0:
            by_task[row["task_id"], row["policy"]][row["trajectory_id"]][row["turn"]] = row

    tasks = defaultdict(list)
    pair_count = 0
    aligned_count = 0
    for (task, _), grouped in by_task.items():
        for a, b in combinations(grouped.values(), 2):
            common = a.keys() & b.keys()
            if not common:
                continue
            representative = min(common)
            if a[representative]["outcome"] == b[representative]["outcome"]:
                continue
            win, lose = (a, b) if a[representative]["outcome"] else (b, a)
            wins = []
            for key in PREDICTORS[:2]:
                delta = np.array([win[i][key] - lose[i][key] for i in sorted(common)])
                wins.append(float(np.mean((delta > 1e-6) + 0.5 * (np.abs(delta) <= 1e-6))))
            tasks[task].append(wins)
            pair_count += 1
            aligned_count += len(common)
    result: dict[str, int | dict] = {
        "tasks": len(tasks),
        "attempt_pairs": pair_count,
        "matched_turn_pairs": aligned_count,
    }
    if tasks:
        task_means = np.array([np.mean(attempts, axis=0) for attempts in tasks.values()])
        ci = np.array(bootstrap_means(np.ones((len(tasks), 1)), task_means))
        for i, key in enumerate(PREDICTORS[:2]):
            result[key] = {"concordance": float(task_means[:, i].mean()), "ci95": ci[:, i].tolist()}
    return result


def summarize(rows: list[dict]) -> dict:
    y = np.array([r["outcome"] for r in rows])
    return {
        "tasks": len({r["task_id"] for r in rows}),
        "trajectories": len({r["trajectory_id"] for r in rows}),
        "turns": len(rows),
        "success_fraction": float(y.mean()),
        **{key: scores(y, np.array([r[key] for r in rows])) for key in PREDICTORS},
    }


def report(rows: list[dict], output: Path) -> dict:
    """Write held-out reliability plots and compute pooled and sliced statistics."""
    if not rows:
        raise ValueError("no held-out turns to evaluate")
    curves = {}
    y = np.array([row["outcome"] for row in rows])
    figure = Figure(figsize=(6, 6), layout="constrained")
    axis = figure.subplots()
    axis.plot((0, 1), (0, 1), color="gray", linestyle="--", linewidth=1)
    for key, label in (("value", "Value model"), ("task_only", "Task only"), ("global_rate", "Global rate")):
        curve = calibration(y, np.array([r[key] for r in rows]))
        curves[key] = curve
        axis.plot(
            [point["mean_prediction"] for point in curve],
            [point["observed_rate"] for point in curve],
            marker="o",
            label=label,
        )
    axis.set(xlabel="Mean predicted probability", ylabel="Observed success frequency", xlim=(0, 1), ylim=(0, 1))
    axis.set_title("Held-out task calibration")
    axis.legend()
    figure.savefig(output, dpi=160)
    slices = defaultdict(list)
    for row in rows:
        length = row["total_turns"]
        bucket = "2-20" if length <= 20 else "21-50" if length <= 50 else "51+"
        slices["trajectory_length", bucket].append(row)
        slices["outcome", str(row["outcome"])].append(row)
        slices["policy", row["policy"]].append(row)
        slices["task_family", row["family"]].append(row)
        slices["repository", row["repository"]].append(row)
        slices["decision", "initial" if row["turn"] == 0 else "after action"].append(row)
        if row["turn"] == row["total_turns"] - 1:
            slices["decision", "final"].append(row)
    sliced: dict = defaultdict(dict)
    for (name, key), group in slices.items():
        sliced[name][key] = summarize(group)
    return {
        "overall": summarize(rows),
        "paired_value_minus_task_only": paired_difference(rows, "task_only"),
        "paired_value_minus_global": paired_difference(rows, "global_rate"),
        "within_task": matched_turns(rows),
        "calibration": curves,
        "slices": dict(sliced),
    }
