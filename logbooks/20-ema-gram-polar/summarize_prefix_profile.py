# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize regional profile evidence without equating device time with wall time."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(report, manifest):
    assert report["completed"] and len(report["hosts"]) == 4
    assert report["manifest"] == manifest
    hosts = []
    for index, host in enumerate(report["hosts"]):
        assert host["profile"] == manifest["profiles"][index]
        identity = host["verified_identity"]
        assert identity["bytes"] == host["profile"]["bytes"]
        assert identity["md5_hash"] == host["profile"]["md5_hash"]
        data = host["summary"]
        assert data["trace_provenance"]["trace_sha256"] == identity["sha256"]
        regions = {row["path"]: row for row in data["hierarchical_regions"]}
        root = regions["_train_step"]["inclusive_duration"]
        power = regions["_train_step=>gram_direction=>relative_gram_power"]["inclusive_duration"]
        gathers = [
            row
            for row in data["hot_ops"]
            if row["canonical_name"] == "all-gather" and row["shape_signature"] == "16,4096,4096"
        ]
        assert len(gathers) == 3 and {row["name"] for row in gathers} == {
            "all-gather.62",
            "all-gather.74",
            "all-gather.88",
        }
        hosts.append(
            dict(
                host_index=index,
                profile_identity=identity,
                trace_overview=data["trace_overview"],
                parsed_step_markers=data["step_time"]["all_steps"],
                named_train_device_time_seconds=root / 1e6,
                named_relative_power_device_time_seconds=power / 1e6,
                relative_power_share_of_named_train_device_time=power / root,
                large_all_gathers=gathers,
                per_track_breakdown=data["time_breakdown"],
                parser_peak_rss_gib=host["parser_peak_rss_kib"] / 1024**2,
                full_summary_uri=host["full_summary_uri"],
                full_summary_identity=host["full_summary_identity"],
            )
        )
    return dict(
        source_commit=report["source_commit"],
        hosts=hosts,
        limits=[
            "One profile window around native loss label100; not the designated800..899steadywindow.",
            "Per-track sums include overlapping devices/tracks; hierarchical parent and child regions overlap.",
            "Named relative-power share is device-op attribution, not a measured wall-clock Amdahl fraction.",
            "Parser step markers are not training-update counts; do not use their mean/median for training MFU.",
            "All-gather event durations can contain arrival/wait imbalance; they do not isolate network transfer time.",
            "Source padding11to16 is known, but these aggregates do not map padded slots to devices or prove a cause.",
            "No numerical change, speedup forecast or training-quality acceptance follows from this report alone.",
        ],
        next_decision=(
            "Inspect exact per-device XPlane event intervals, source scopes and matrix-slot placement regionally; "
            "Distinguish eigensolver compute from collective arrival/wait "
            "before selecting a change that preserves geometry. "
            "Continue the924update loss experiment unchanged."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    result = summarize(json.loads(raw), json.loads(args.manifest.read_bytes()))
    result["report_identity"] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
