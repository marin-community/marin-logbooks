# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recompute variant scores from a complete scoring run without model calls."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from targets import Target, targets


def target_key(item: dict) -> tuple:
    return tuple(item[key] for key in ("variant", "method", "strand", "mapping", "candidate"))


def expected_key(target: Target) -> tuple:
    return (target.variant, target.method, target.strand, target.mapping, target.candidate)


def aggregate(panels_path: Path, scores_path: Path) -> list[dict]:
    panels_bytes = panels_path.read_bytes()
    panels = json.loads(panels_bytes)
    records = [json.loads(line) for line in scores_path.read_text().splitlines()]
    assert records[0]["type"] == "start" and records[-1]["type"] == "complete", "Run is incomplete"
    header = records[0]
    assert header["protocol_version"] == panels["protocol_version"] == 3
    assert header["panels_sha256"] == hashlib.sha256(panels_bytes).hexdigest()
    rows = panels["panels"][header["panel"]][: header["variant_count"]]
    expected = {expected_key(target): target for target in targets(rows, panels["panels"]["demonstrations"])}
    scores = {}
    for record in records[1:-1]:
        assert record["type"] == "score"
        key = target_key(record)
        assert key in expected and key not in scores
        target = expected[key]
        assert record["input_sha256"] == hashlib.sha256((target.prefix + target.text).encode()).hexdigest()
        assert math.isfinite(record["logprob"])
        scores[key] = record["logprob"]
    assert set(scores) == set(expected), "Missing scoring targets"
    output = []
    for row in rows:
        item = {key: row[key] for key in ("id", "label", "subset", "match_group", "chrom")}
        item["answer_mapping"] = row["answer_mapping"]
        for method in ("null", "healthy", "zero-shot", "two-shot"):
            strands = {}
            for strand in ("+", "-"):
                if method in {"null", "healthy"}:
                    ref = scores[(row["id"], method, strand, "none", "reference")]
                    alt = scores[(row["id"], method, strand, "none", "alternate")]
                    strands[strand] = ref - alt
                else:
                    mapping = row["answer_mapping"]
                    pathogenic, benign = ("A", "B") if mapping == "primary" else ("B", "A")
                    strands[strand] = (
                        scores[(row["id"], method, strand, mapping, pathogenic)]
                        - scores[(row["id"], method, strand, mapping, benign)]
                    )
            item[method] = (strands["+"] + strands["-"]) / 2
            item[f"{method}/strand-difference"] = strands["+"] - strands["-"]
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.panels, args.scores)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
