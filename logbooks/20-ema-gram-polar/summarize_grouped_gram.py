# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and join four bounded grouped-Gram reports using only static JSON."""

import argparse
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path

from summarize_saved_state_check import CASES
from summarize_saved_state_check import summarize as join_saved_state

LAYOUTS = ("matrix", "matrix_grouped")


def summarize(reports, manifest, checkpoint_identity, source_commit, *, processes=4, devices=16):
    assert len(reports) == processes and len(source_commit) == 40
    converted = []
    timings = {case: {} for case in CASES}
    for report in reports:
        assert report["source_commit"] == source_commit
        assert report["reference"] == LAYOUTS[0] and report["candidate"] == LAYOUTS[1]
        assert report["samples"] == 3 and report["training_steps"] == 0
        assert report["full_training_memory_fit"] is False
        assert report["full_trainer_restore"] is False
        adapted = copy.deepcopy(report)
        # The source-bound harness fingerprints inputs through snapshot(), which
        # rejects nonfinite shards; finite/input_immutable flags cover both cases.
        # Adapt only field names for the existing strict leaf/index join.
        adapted.update(
            input_state_finite=True,
            training_steps_executed=0,
            full_training_memory_fit_verified=False,
        )
        for case in CASES:
            item = report["cases"][case]
            assert set(item) == {*LAYOUTS, "comparison"}
            adapted_case = {"comparison": item["comparison"]}
            process_times = {}
            for layout, legacy_name in zip(LAYOUTS, ("replicated", "matrix"), strict=True):
                row = item[layout]
                assert row["finite"] is True and row["input_immutable"] is True
                first = row["compile_and_first_seconds"]
                samples = row["warmed_sample_seconds"]
                assert len(samples) == report["samples"]
                assert all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in [first, *samples])
                replay = row["replay_comparisons"]
                assert len(replay) == report["samples"] and all(type(v) is bool for v in replay)
                assert type(row["bitwise_replays"]) is int and row["bitwise_replays"] == sum(replay)
                process_times[layout] = dict(
                    compile_and_first_seconds=first,
                    warmed_sample_seconds=samples,
                    mean_seconds=statistics.mean(samples),
                    median_seconds=statistics.median(samples),
                    min_seconds=min(samples),
                    max_seconds=max(samples),
                    bitwise_replays=row["bitwise_replays"],
                    replay_comparisons=replay,
                )
                adapted_case[legacy_name] = dict(
                    all_local_outputs_finite=row["finite"],
                    input_state_unchanged=row["input_immutable"],
                    seconds_including_compilation=first,
                )
            process_times["baseline_over_candidate_mean_ratio"] = (
                process_times["matrix"]["mean_seconds"] / process_times["matrix_grouped"]["mean_seconds"]
            )
            timings[case][str(report["process_index"])] = process_times
            adapted["cases"][case] = adapted_case
        converted.append(adapted)
    result = join_saved_state(converted, manifest, checkpoint_identity, processes=processes, devices=devices)
    for case in CASES:
        result["cases"][case].pop("seconds_including_compilation_by_process")
        result["cases"][case]["timing_by_process"] = timings[case]
        result["cases"][case]["timing_scope"] = (
            "Three locally synchronized optimizer samples, fixed matrix then matrix_grouped order; "
            "compilation/first execution separate. Host durations are not summed. "
            "No training throughput, MFU, independent seeds or timing-order control."
        )
    result.update(reference=LAYOUTS[0], candidate=LAYOUTS[1], source_commit=source_commit)
    result["limitations"] = (
        "Static report completeness/consistency only. Canonical job/task success, "
        "same-attempt provenance and online logging require separate verification. "
        "Finite tolerance failures and non-bitwise replays remain diagnostics."
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).parent
    result = summarize(
        [json.loads(path.read_text()) for path in args.reports],
        json.loads((base / "grouped-gram-manifest.json").read_text()),
        json.loads((base / "loss-pair-checkpoint-metadata.json").read_text()),
        json.loads((base / "grouped-gram-submitted.json").read_text())["source_commit"],
    )
    result["input_reports"] = [
        dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in args.reports
    ]
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
