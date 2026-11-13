# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Describe prompt sensitivity and matched-control failures without model calls."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

METHODS = ("null", "healthy", "zero-shot", "two-shot")


def difference_summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean_signed": statistics.mean(values),
        "mean_absolute": statistics.mean(abs(value) for value in values),
        "max_absolute": max(abs(value) for value in values),
    }


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["match_group"]].append(row)
    assert all(len(group) == 10 and sum(row["label"] for row in group) == 1 for group in groups.values())
    result = {"variants": len(rows), "groups": len(groups), "strand": {}, "mapping": {}, "positive_ranks": {}}
    for method in METHODS:
        result["strand"][method] = difference_summary([row[f"{method}/strand-difference"] for row in rows])
        ranks = []
        for group_id, group in groups.items():
            positive = next(row for row in group if row["label"])
            controls = [row for row in group if not row["label"]]
            rank = 1 + sum(row[method] > positive[method] for row in controls)
            rank += 0.5 * sum(row[method] == positive[method] for row in controls)
            ranks.append(
                {
                    "match_group": group_id,
                    "positive": positive["id"],
                    "rank": rank,
                    "score": positive[method],
                    "top_control": max(controls, key=lambda row: row[method])["id"],
                }
            )
        result["positive_ranks"][method] = sorted(ranks, key=lambda row: (-row["rank"], row["match_group"]))
    result["mapping_assignment_counts"] = {
        mapping: {
            "variants": sum(row["answer_mapping"] == mapping for row in rows),
            "positives": sum(row["answer_mapping"] == mapping and row["label"] for row in rows),
        }
        for mapping in ("primary", "reversed")
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.variant_scores.read_text()))
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
