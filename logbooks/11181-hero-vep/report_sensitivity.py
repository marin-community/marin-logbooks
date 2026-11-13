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
    for method in ("zero-shot", "two-shot"):
        summary = difference_summary([row[method] - row[f"{method}/reversed"] for row in rows])
        summary["strict_sign_flips"] = sum(row[method] * row[f"{method}/reversed"] < 0 for row in rows)
        flips = ties = pairs = 0
        for group in groups.values():
            positive = next(row for row in group if row["label"])
            for control in (row for row in group if not row["label"]):
                primary = positive[method] - control[method]
                reversed_delta = positive[f"{method}/reversed"] - control[f"{method}/reversed"]
                flips += primary * reversed_delta < 0
                ties += primary == 0 or reversed_delta == 0
                pairs += 1
        summary.update(matched_pairs=pairs, strict_pair_order_flips=flips, pairs_with_tie=ties)
        result["mapping"][method] = summary
    format_rows = [row for row in rows if "explicit-pair" in row]
    if format_rows:
        result["format"] = difference_summary([row["explicit-pair"] - row["zero-shot"] for row in format_rows])
        result["format"]["strict_sign_flips"] = sum(row["explicit-pair"] * row["zero-shot"] < 0 for row in format_rows)
        result["format"]["variant_ids"] = [row["id"] for row in format_rows]
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
