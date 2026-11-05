# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the predeclared matched-seed/task intervals to compact study results."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import t


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    confidence: float
    seeds: int
    seed_variance: float
    task_variance: float = 0.0


def quality_interval(
    difference: np.ndarray, families: np.ndarray, weights: dict[str, float], confidence: float
) -> Interval:
    """Estimate a macro success difference with conservative crossed seed/task variance."""
    seeds, tasks = difference.shape
    if seeds < 3 or len(families) != tasks or not np.isfinite(difference).all():
        raise ValueError("Quality inference requires complete finite paired seeds and tasks")
    if set(families) != set(weights) or not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Task families must match the predeclared normalized weights")
    per_seed = np.zeros(seeds)
    per_task = difference.mean(axis=0)
    task_variance = 0.0
    for family, weight in weights.items():
        mask = families == family
        if mask.sum() < 2:
            raise ValueError(f"Too few tasks in {family}")
        per_seed += weight * difference[:, mask].mean(axis=1)
        task_variance += weight**2 * per_task[mask].var(ddof=1) / mask.sum()
    seed_variance = float(per_seed.var(ddof=1) / seeds)
    standard_error = np.sqrt(seed_variance + task_variance)
    radius = t.ppf((1 + confidence) / 2, df=seeds - 1) * standard_error
    estimate = float(per_seed.mean())
    return Interval(
        estimate,
        float(estimate - radius),
        float(estimate + radius),
        confidence,
        seeds,
        seed_variance,
        float(task_variance),
    )


def cost_interval(log_ratios: np.ndarray, confidence: float) -> Interval:
    """Estimate the relative training-cycle cost from paired seed log ratios."""
    seeds = len(log_ratios)
    if seeds < 3 or not np.isfinite(log_ratios).all():
        raise ValueError("Cost inference requires at least three finite paired seeds")
    variance = float(log_ratios.var(ddof=1) / seeds)
    radius = t.ppf((1 + confidence) / 2, df=seeds - 1) * np.sqrt(variance)
    center = float(log_ratios.mean())
    return Interval(
        float(np.expm1(center)),
        float(np.expm1(center - radius)),
        float(np.expm1(center + radius)),
        confidence,
        seeds,
        variance,
    )


def decision(quality: Interval, cost: Interval) -> str:
    if quality.low > -0.01 and cost.high < 0.05:
        return "acceptable trade-off within the predeclared margins"
    if quality.high < -0.01 or cost.low > 0.05:
        return "material regression beyond a predeclared margin"
    return "inconclusive"


def analyze(path: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    weights = dict(zip(protocol["families"], protocol["primary_metric"]["family_weights"], strict=True))
    with np.load(path, allow_pickle=False) as data:
        seeds, arms = data["seeds"].tolist(), data["arms"].tolist()
        tasks, families = data["tasks"], data["families"]
        success, scores = data["success"].astype(float), data["scores"].astype(float)
        seconds = data["training_seconds"].astype(float)
        updates = data["updates"]
    allowed = protocol["initial_seeds"] + protocol["extension"]["maximum_additional_matched_seeds"]
    if seeds != allowed[: len(seeds)] or not 3 <= len(seeds) <= 6:
        raise ValueError("Seeds differ from the fixed initial study and bounded extension")
    if set(arms) != {"tis", "sc128", "sc32"} or len(arms) != 3:
        raise ValueError("All three predeclared arms are required")
    quality_rows = protocol["evaluation"].get("quality_rows", protocol["split"]["heldout_rows"])
    if len(tasks) != len(set(tasks.tolist())) or len(tasks) != quality_rows:
        raise ValueError("The complete unique held-out task panel is required")
    if success.shape != (len(seeds), 3, len(tasks)) or scores.shape != success.shape:
        raise ValueError("The seed/arm/task matrix is incomplete")
    if not np.isfinite(success).all() or not np.isfinite(scores).all() or not np.isin(success, (0, 1)).all():
        raise ValueError("Missing or invalid grades cannot become zero-success observations")
    if seconds.shape != (len(seeds), 3) or not np.isfinite(seconds).all() or not (seconds > 0).all():
        raise ValueError("Every seed/arm needs a positive complete-cycle measurement")
    if updates.shape != seconds.shape or not (updates == protocol["training"]["planned_updates"]).all():
        raise ValueError("All arms must reach the same predeclared policy-update endpoint")
    family_counts = protocol["evaluation"].get(
        "quality_per_family", {family: protocol["split"]["heldout_per_family"] for family in weights}
    )
    if any((families == family).sum() != family_counts[family] for family in weights):
        raise ValueError("The held-out family denominators changed")

    positions = {arm: arms.index(arm) for arm in arms}
    comparisons = {}
    for left, right in (("sc32", "tis"), ("sc128", "tis"), ("sc32", "sc128")):
        difference = success[:, positions[left]] - success[:, positions[right]]
        log_ratios = np.log(seconds[:, positions[left]] / seconds[:, positions[right]])
        intervals = {
            str(level): {
                "quality_difference": asdict(quality_interval(difference, families, weights, level)),
                "relative_training_cycle_cost": asdict(cost_interval(log_ratios, level)),
            }
            for level in (0.95, 0.975)
        }
        quality = quality_interval(difference, families, weights, 0.975)
        cost = cost_interval(log_ratios, 0.975)
        comparisons[f"{left}_versus_{right}"] = {
            "intervals": intervals,
            "two_look_adjusted_decision": decision(quality, cost),
            "paired_seed_cost_ratios": np.expm1(log_ratios).tolist(),
            "arithmetic_ratio_of_mean_times": float(
                seconds[:, positions[left]].mean() / seconds[:, positions[right]].mean() - 1
            ),
        }
    per_seed = []
    for seed_index, seed in enumerate(seeds):
        for arm_index, arm in enumerate(arms):
            family_results = {
                family: {
                    "tasks": int((families == family).sum()),
                    "full_success": float(success[seed_index, arm_index, families == family].mean()),
                    "mean_verifier_reward": float(scores[seed_index, arm_index, families == family].mean()),
                }
                for family in weights
            }
            per_seed.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "families": family_results,
                    "macro_full_success": sum(
                        weights[family] * row["full_success"] for family, row in family_results.items()
                    ),
                    "training_seconds": float(seconds[seed_index, arm_index]),
                    "seconds_per_update": float(seconds[seed_index, arm_index] / updates[seed_index, arm_index]),
                }
            )
    return {
        "stage": "initial" if len(seeds) == 3 else "extension",
        "seeds": seeds,
        "tasks": len(tasks),
        "curve_quality_tasks": protocol["evaluation"].get("curve_quality_rows", 1000),
        "comparisons": comparisons,
        "per_seed": per_seed,
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "task_scope_note": protocol["uncertainty"].get("post_outcome_scope_amendment"),
        "units": "Quality differences are fractions, not percentage points; relative cost is ratio minus one.",
        "limits": (
            "Small-sample t intervals under the predeclared crossed seed/task model. "
            "Task contribution conservatively retains interaction variance. "
            "Two-look coverage is marginal per metric/comparison, not joint across every result."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    arguments = parser.parse_args()
    arguments.output.write_text(json.dumps(analyze(arguments.input, arguments.protocol), indent=2) + "\n")
