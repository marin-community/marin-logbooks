# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reject incomplete or misidentified interval reports using the saved result."""

import argparse
import copy
import json
from pathlib import Path

from summarize_prefix_intervals import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    prior = json.loads(args.prior.read_text())
    manifest = json.loads(args.manifest.read_text())
    assert len(summarize(report, manifest, prior)["hosts"]) == 4
    rejected = []
    for label in (
        "incomplete",
        "source",
        "manifest",
        "sha",
        "missing_track",
        "duplicate_event",
        "duration",
        "shape",
        "scope",
    ):
        value = copy.deepcopy(report)
        h = value["hosts"][0]
        e = h["summary"]["selected_collectives"][0]
        if label == "incomplete":
            value["completed"] = False
        elif label == "source":
            value["source_commit"] = "wrong"
        elif label == "manifest":
            value["manifest_sha256"] = "wrong"
        elif label == "sha":
            h["verified_identity"]["sha256"] = "wrong"
        elif label == "missing_track":
            h["summary"]["tracks"].pop()
        elif label == "duplicate_event":
            h["summary"]["selected_collectives"][1] = copy.deepcopy(e)
        elif label == "duration":
            e["duration_ps"] += 1
        elif label == "shape":
            e["stats"]["shape_with_layout"] = "f16[16,4096,4096]"
        elif label == "scope":
            e["tf_op"] = "unrelated"
        try:
            summarize(value, manifest, prior)
        except AssertionError:
            rejected.append(label)
        else:
            raise AssertionError(label)
    print(json.dumps(dict(passed=True, rejected_mutations=rejected, actual_report_summary_passed=True)))


if __name__ == "__main__":
    main()
