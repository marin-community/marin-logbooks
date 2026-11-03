# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize declared slice ranges without interpreting ConcatBitcast output semantics."""

import argparse
import hashlib
import json
import re
from itertools import pairwise
from pathlib import Path

from compare_down_concat import compare, indexed
from trace_down_concat import ROLE
from trace_start_producers import identity


def unique_signature_mapping(values, reference):
    result = {}
    for key, value in values.items():
        found = [k for k, base in reference.items() if base["normalized_signature"] == value["normalized_signature"]]
        assert len(found) == 1
        result[int(key.rsplit("/", 1)[1])] = int(found[0].rsplit("/", 1)[1])
    assert sorted(result) == [0, 1, 2, 3] and sorted(result.values()) == [0, 1, 2, 3]
    return result


def summarize(report, inventory):
    comparisons = compare(report)["graphs"]
    result = {}
    for graph, artifact in report["artifacts"].items():
        assert artifact["input_sha256"] == inventory["manifest"]["artifacts"][graph]["sha256"]
        polar = {
            x["operation"]: (group, x["iteration"], x["product"])
            for group, chain in inventory["artifacts"][graph]["polar_lineage"]["output_groups"].items()
            for x in chain
        }
        routes, producer_ids = [], set()
        for key, value in sorted(indexed(artifact).items()):
            assert len(value["boundaries"]) == 1
            producer = value["records"][value["boundaries"][0]["node"]]
            assert producer["opcode"] in ("dot", "convolution")
            assert polar[producer["computation"] + "/" + producer["name"]] == ("down_proj", 5, "X_Xt")
            producer_ids.add(identity(producer))
            slices = [row for row in value["records"] if row["opcode"] == "slice-start"]
            assert len(slices) == 1
            match = re.search(r"slice=\{\[(\d+):(\d+)\], \[0:128\], \[0:1024\]\}", slices[0]["attributes"])
            assert match
            converts = [row for row in value["records"] if row["opcode"] == "convert"]
            assert len(converts) == 1
            routes.append(
                dict(
                    operand_position=int(key.rsplit("/", 1)[1]),
                    leading_axis_interval=list(map(int, match.groups())),
                    slice_start=slices[0],
                    conversion=converts[0],
                    product=producer,
                )
            )
        assert len(producer_ids) == 1
        intervals = sorted(row["leading_axis_interval"] for row in routes)
        assert intervals[0][0] == 0 and intervals[-1][1] == 11
        assert all(a[1] == b[0] for a, b in pairwise(intervals))
        result[graph] = dict(
            role=ROLE,
            routes=routes,
            same_product_identity_within_graph=True,
            unique_full_signature_mapping_to_untouched=unique_signature_mapping(
                comparisons[graph], comparisons["matrix_untouched"]
            ),
        )
    return dict(
        graphs=result,
        interpretation=(
            "Mapping compares complete normalized declared input-route signatures, preserving original operand order. "
            "It does not permute any executable or establish ConcatBitcast semantics, runtime values, order or cause. "
            "Both full28leaf numerical substitution gates remain failed."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw, inventory = args.report.read_bytes(), args.inventory.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.sha256
    assert hashlib.sha256(inventory).hexdigest() == args.inventory_sha256
    args.output.write_text(json.dumps(summarize(json.loads(raw), json.loads(inventory)), indent=2) + "\n")


if __name__ == "__main__":
    main()
