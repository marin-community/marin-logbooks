# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regional, streaming S0 counterfactual and bounded deterministic review export."""

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict

import numpy as np
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.redaction import redact_string
from rolloutengine.models import ATTEMPT_SECONDS_METRIC

from vendor.skyrl.scripts.loop_audit import read_rows, trial_turns, verdict_passed
from vendor.skyrl.scripts.repeat_stall_counterfactual import simulate_trial, summarize


def bounded(text, limit=2400):
    clean = redact_string(text)
    if len(clean) <= limit:
        return clean
    return clean[: limit // 2] + "\n[bounded excerpt]\n" + clean[-limit // 2 :]


def input_rows(pattern, replacements, membership):
    for row in read_rows(pattern):
        if membership.get(row.get("task_id")) not in replacements:
            yield row
    for cohort, replacement in replacements.items():
        for row in read_rows(replacement):
            if membership.get(row.get("task_id")) != cohort:
                raise ValueError("Replacement input contains a task outside its frozen cohort")
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument(
        "--replacement-input",
        action="append",
        default=[],
        help="COHORT=JSONL_URI for a recorded infrastructure recovery",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    manifest = json.loads(StoragePath(args.manifest).read_text())
    replacements = {}
    for value in args.replacement_input:
        cohort, pattern = value.split("=", 1)
        if cohort in replacements or cohort not in manifest["cohorts"]:
            raise ValueError("Duplicate or unknown replacement cohort")
        replacements[cohort] = pattern
    membership = {}
    for cohort, definition in manifest["cohorts"].items():
        for entry in definition["rows"]:
            task_id = entry["task_id"]
            if task_id in membership:
                raise ValueError("Task ID belongs to multiple cohorts")
            membership[task_id] = cohort
    trials = defaultdict(list)
    suppressed = defaultdict(int)
    timing = defaultdict(list)
    selected = []
    for ordinal, row in enumerate(input_rows(args.pattern, replacements, membership)):
        cohort = membership.get(row.get("task_id"))
        if cohort is None:
            continue
        row = dict(row)
        if row.get("ordinal") is None:
            row["ordinal"] = ordinal
        simulation, waits = simulate_trial(row, 3)
        trials[cohort].append(simulation)
        suppressed[cohort] += waits
        duration = (row.get("env_metrics") or {}).get(ATTEMPT_SECONDS_METRIC)
        if verdict_passed(row) is not None and isinstance(duration, (int, float)):
            timing[cohort].append(float(duration))
        if not simulation["fired"]:
            continue
        identity = [cohort, *[simulation.get(key) for key in ("task_id", "attempt_id", "uid", "ordinal")]]
        rank = hashlib.sha256(json.dumps([20261125, identity], separators=(",", ":")).encode()).hexdigest()
        if len(selected) == 30 and rank >= selected[-1]["sample_rank"]:
            continue
        turns = trial_turns(row.get("messages") or ())
        fire = simulation["fire_index"]
        window = []
        for index in range(max(0, fire - 3), min(len(turns), fire + 3)):
            turn = asdict(turns[index])
            turn["index"] = index
            for key in ("action", "observation", "message", "keystrokes"):
                value = turn[key]
                turn[key + "_sha256"] = hashlib.sha256(value.encode()).hexdigest()
                turn[key] = bounded(value)
            window.append(turn)
        selected.append({"cohort": cohort, "sample_rank": rank, **simulation, "window": window})
        selected.sort(key=lambda item: item["sample_rank"])
        selected = selected[:30]
    reports = []
    for cohort, definition in manifest["cohorts"].items():
        expected = len(definition["rows"]) * definition.get("trials_per_task", 1)
        if len(trials[cohort]) != expected:
            raise ValueError(f"{cohort}: expected {expected} saved trials, found {len(trials[cohort])}")
        reports.append(summarize(cohort, trials[cohort], suppressed[cohort], 3, 823))
    destination = StoragePath(args.output)
    destination.mkdirs()
    (destination / "counterfactual.json").write_text(json.dumps({"batches": 3, "cohorts": reports}, indent=2) + "\n")
    (destination / "review-sample.json").write_text(
        json.dumps(
            {
                "seed": 20261125,
                "selection": (
                    "30 smallest SHA256(seed, cohort, task_id, attempt_id, uid, ordinal), "
                    "independent of cut length and verdict"
                ),
                "population_fired": sum(report["fired_trials"] for report in reports),
                "sample_n": len(selected),
                "trials": selected,
            },
            indent=2,
        )
        + "\n"
    )
    statistics = {}
    for cohort, values in timing.items():
        statistics[cohort] = {
            "graded_timing_n": len(values),
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "max": max(values),
        }
    (destination / "graded-timing.json").write_text(json.dumps(statistics, indent=2) + "\n")
    print(
        json.dumps(
            {
                "reports": [
                    {key: value for key, value in report.items() if key != "hand_check_sample"} for report in reports
                ],
                "review_sample_n": len(selected),
                "timing": statistics,
            }
        )
    )


if __name__ == "__main__":
    main()
