# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize one reconciled public-50 solver repetition, including judge recoveries."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

MANIFEST = Path(__file__).with_name("task-manifest.json")
SCORED = {"scored", "recovered_from_reports", "recovered_from_trajectory"}


def summarize_pass(scorecard: Path) -> dict:
    manifest = {t["task"]: "derived" if t["input_variant"] else "unchanged" for t in json.loads(MANIFEST.read_text())}
    with scorecard.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 50 or {row["task"] for row in rows} != set(manifest):
        raise ValueError("Expected exactly one row for each public task")
    for row in rows:
        if row["input_group"] != manifest[row["task"]]:
            raise ValueError(f"Incorrect input group for {row['task']}")
        for field in ("signed_score", "unsigned_score"):
            raw = row[field]
            if bool(raw) != (row["final_state"] in SCORED) or (raw and not 0 <= int(raw) <= 100):
                raise ValueError(f"Inconsistent {field} and outcome for {row['task']}")

    result = {}
    for group in ("all", "unchanged", "derived"):
        selected = [row for row in rows if group == "all" or row["input_group"] == group]
        graded = [row for row in selected if row["final_state"] in SCORED]
        result[group] = {
            "tasks": len(selected),
            "graded": len(graded),
            "outcomes": dict(sorted(Counter(row["final_state"] for row in selected).items())),
            "signed_mean": mean(int(row["signed_score"]) for row in graded) if len(graded) == len(selected) else None,
            "unsigned_mean": mean(int(row["unsigned_score"]) for row in graded) if len(graded) == len(selected) else None,
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scorecard", type=Path)
    print(json.dumps(summarize_pass(parser.parse_args().scorecard), indent=2))
