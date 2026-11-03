# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the selected ladder's sampled Paloma time-to-loss evidence."""

import argparse
import concurrent.futures
import json
import math
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import wandb

PROJECT = "marin-community/marin_moe"
CAMPAIGN = "hero-dense-opt-20261026"
METRIC = "eval/paloma/macro_loss"
THRESHOLDS = (4.0, 3.75, 3.5, 3.25, 3.1, 3.0, 2.9)
SELECTED = (("adamw", 3), ("adamh", 3), ("muon", 3), ("muonh", 1))


def read_run(work: tuple[str, int, str, int]) -> dict[str, Any]:
    budget, updates, optimizer, multiplier = work
    run_id = f"{CAMPAIGN}-{budget}-{optimizer}-lr{multiplier}"
    run = wandb.Api(timeout=30).run(f"{PROJECT}/{run_id}")
    return {
        "budget": budget,
        "updates": updates,
        "optimizer": optimizer,
        "lr_multiplier": multiplier,
        "id": run_id,
        "state": run.state,
        "created_at": run.created_at,
        "first_training_rows": list(
            run.scan_history(keys=["_step", "_timestamp", "train/loss"], min_step=0, max_step=5, page_size=10)
        ),
        "evaluations": list(
            run.scan_history(keys=["_step", "_timestamp", METRIC], min_step=0, max_step=updates, page_size=200000)
        ),
    }


def fetch_histories(plan: dict[str, Any]) -> dict[str, Any]:
    work = [
        (f"{rung['budget']:.0e}", rung["steps"], optimizer, multiplier)
        for rung in plan["rungs"]
        for optimizer, multiplier in SELECTED
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(read_run, work))
    return {"observed_utc": datetime.now(UTC).isoformat(), "runs": runs}


def derive_run(run: dict[str, Any]) -> dict[str, Any]:
    first = min(run["first_training_rows"], key=lambda row: row["_step"])
    assert first["_step"] == 0
    start = first["_timestamp"]
    evaluations = sorted(run["evaluations"], key=lambda row: row["_step"])
    assert run["state"] == "finished"
    assert evaluations[-1]["_step"] == run["updates"] - 1
    assert all(math.isfinite(row[METRIC]) and row["_timestamp"] >= start for row in evaluations)
    assert all(a["_timestamp"] < b["_timestamp"] for a, b in pairwise(evaluations))
    derived = {
        key: value
        for key, value in run.items()
        if key not in {"thresholds", "first_logged_training_utc", "final_evaluation_elapsed_wall_hours"}
    }
    derived["first_logged_training_utc"] = datetime.fromtimestamp(start, UTC).isoformat()
    derived["final_evaluation_elapsed_wall_hours"] = (evaluations[-1]["_timestamp"] - start) / 3600
    thresholds = []
    for threshold in THRESHOLDS:
        index = next((i for i, row in enumerate(evaluations) if row[METRIC] <= threshold), None)
        if index is None:
            thresholds.append({"threshold": threshold, "observed_hit": False})
            continue
        hit = evaluations[index]
        previous = evaluations[index - 1] if index else None
        thresholds.append(
            {
                "threshold": threshold,
                "observed_hit": True,
                "first_observed_hit_step": hit["_step"],
                "first_observed_hit_loss": hit[METRIC],
                "first_observed_hit_utc": datetime.fromtimestamp(hit["_timestamp"], UTC).isoformat(),
                "first_observed_hit_elapsed_wall_hours": (hit["_timestamp"] - start) / 3600,
                "previous_sample_step": previous["_step"] if previous else None,
                "previous_sample_loss": previous[METRIC] if previous else None,
                "previous_sample_elapsed_wall_hours": (previous["_timestamp"] - start) / 3600 if previous else None,
            }
        )
    derived["thresholds"] = thresholds
    return derived


def analyze(history: dict[str, Any]) -> dict[str, Any]:
    runs = [derive_run(run) for run in history["runs"]]
    assert len(runs) == 28 and len({run["id"] for run in runs}) == 28
    method = {
        "clock": "UTC elapsed hours from the first logged training row at zero-based step 0 to an " "evaluation row",
        "includes": (
            "All intervening wall time, including evaluations, checkpointing, retries, preemption " "waits and replay"
        ),
        "excludes": "Initial scheduling, process startup and compilation before the first logged step 0",
        "thresholds": list(THRESHOLDS),
        "threshold_selection": (
            "Descriptive round loss levels selected after observing the completed " "ladder; not preregistered"
        ),
        "sampling": "All recorded Paloma evaluations, normally each quarter of updates plus the final " "evaluation",
        "interpretation": (
            "First observed evaluation at or below a threshold, with its previous sampled "
            "evaluation; no interpolation"
        ),
        "bracket_limit": (
            "The previous sample is above threshold when present, but unsampled "
            "non-monotonic crossings are not excluded"
        ),
        "no_hit": "No recorded evaluation met the threshold; unsampled crossings are not excluded",
        "not_compute_time": (
            "This clock is operational latency for this execution, not uninterrupted "
            "optimizer training time or GPU efficiency"
        ),
        "runtime_policy": "W&B _runtime and last-attempt durations are not used",
        "query_bounds": (
            "Exact 28 selected IDs only; initial training steps 0 through 5 and evaluation "
            "history bounded by each target update count"
        ),
    }
    limitations = [
        "One seed per recipe and budget; no uncertainty estimate or statistical significance claim.",
        "LR multipliers transferred from a limited three-point 1e18 calibration; AdamW, AdamH and Muon "
        "selected the upper boundary.",
        "Queueing, preemptions and replay differed across runs, so elapsed-time ordering is not a causal "
        "optimizer speed comparison.",
        "Model size, training token budget and schedule differ across rungs; cross-rung times are not "
        "matched training trajectories.",
        "Sparse evaluation windows can span many hours; exact threshold crossing times and uninterrupted "
        "compute time are not established.",
    ]
    return {
        "campaign": CAMPAIGN,
        "project": PROJECT,
        "metric": METRIC,
        "history_read_utc": history.get("observed_utc", history.get("history_read_utc")),
        "method": method,
        "limitations": limitations,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Archived history or this script's previous evidence output")
    source.add_argument("--refresh", action="store_true", help="Read bounded W&B histories without changing runs")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.refresh:
        history = fetch_histories(json.loads(Path(__file__).with_name("plan.json").read_text()))
    else:
        history = json.loads(args.input.read_text())
    args.output.write_text(json.dumps(analyze(history), indent=2) + "\n")


if __name__ == "__main__":
    main()
