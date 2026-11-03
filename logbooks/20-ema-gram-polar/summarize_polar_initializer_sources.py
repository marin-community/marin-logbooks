# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize declared initializers without assigning semantics to their opaque loop outputs."""

import argparse
import hashlib
import json
from pathlib import Path

from compare_polar_initializers import compare, indexed
from summarize_gram_sources import tuple_leaves

LEFT = "down_proj/left_power/while_initializer/0"
EARLIER = "down_proj/right_power/earlier_while_initializer/0"


def summarize(report):
    comparisons = compare(report)
    result = {}
    for name, graph in report["artifacts"].items():
        values = indexed(graph)
        summaries = {}
        for key in (LEFT, EARLIER):
            value = values[key]
            root = value["records"][value["root"]]
            assert root["opcode"] == "tuple" and len(root["operands"]) == 12 and not root["tuple_path"]
            children = {}
            for edge in value["edges"]:
                children.setdefault(edge["source"], []).append(edge)
            positions = sorted(children[value["root"]], key=lambda e: e["operand_index"])
            assert [e["operand_index"] for e in positions] == list(range(12))
            assert positions[5]["target"] == positions[7]["target"]
            rows = []
            for edge in positions:
                seen, pending = set(), [edge["target"]]
                while pending:
                    node = pending.pop()
                    if node in seen:
                        continue
                    seen.add(node)
                    pending.extend(e["target"] for e in children.get(node, []))
                boundaries = [value["records"][b["node"]] for b in value["boundaries"] if b["node"] in seen]
                rows.append(
                    dict(
                        position=edge["operand_index"], producer=value["records"][edge["target"]], boundaries=boundaries
                    )
                )
            assert [i for i, row in enumerate(rows) if row["boundaries"]] == [3]
            boundaries = rows[3]["boundaries"]
            states = [r for r in boundaries if r["opcode"] == "parameter"]
            products = [r for r in boundaries if r["opcode"] in ("dot", "convolution")]
            assert len(products) == 1 and '/update_gram_state/dot_general"' in products[0]["attributes"]
            expected_states = 0 if key == LEFT and name == "matrix_substitute" else 1
            assert len(states) == expected_states
            if states:
                prefix = (
                    "states"
                    if name == "matrix_untouched"
                    else ("bank[0][1]" if name == "matrix_substitute" else "bank[1]")
                )
                index = 6 if key == LEFT else 5
                assert f'op_name="{prefix}[{index}].left"' in states[0]["attributes"]
            assert not any(r["opcode"] == "while" for r in value["records"])
            summaries[key] = dict(
                original_while_endpoint=graph["trace"]["targets"][key.rsplit("/", 1)[0]]["endpoint"],
                initializer=root,
                positions=rows,
                comparison=comparisons["graphs"][name][key],
            )
        result[name] = summaries
    cap = comparisons["graphs"]["matrix_capture"][LEFT]["rooted_differences"]
    assert len(cap) == 1 and cap[0]["changed_fields"] == ["shape"]
    a, b = tuple_leaves(cap[0]["left"]["shape"]), tuple_leaves(cap[0]["right"]["shape"])
    assert len(a) == len(b) == 6 and [i for i in range(6) if a[i] != b[i]] == [3, 4]
    assert all(x.replace("S(1)", "") == y.replace("S(1)", "") for x, y in zip(a, b, strict=True))
    for name in ("matrix_untouched", "matrix_capture"):
        row = indexed(report["artifacts"][name])[LEFT]["records"][cap[0]["left_node"]]
        assert row["opcode"] == "all-reduce" and row["tuple_path"] == [5]
    earlier = comparisons["graphs"]["matrix_substitute"][EARLIER]["rooted_differences"]
    assert len(earlier) == 1 and earlier[0]["changed_fields"] == ["attributes"]
    assert (
        earlier[0]["left"]["attributes"].replace("channel_id=276", "channel_id=273") == earlier[0]["right"]["attributes"]
    )
    return dict(
        graphs=result,
        limitations=[
            "Endpoint labels identify the consuming dependency, not ownership or semantics of loop values.",
            "Both original loop outputs remain opaque. No loop body or recurrence was traversed.",
            "Whole initializer constituents retain ordered repeated positions and exact callers.",
            "The passing capture sibling-tag mismatch remains; no runtime precision, equality, order or cause claim.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.sha256
    args.output.write_text(json.dumps(summarize(json.loads(raw)), indent=2) + "\n")


if __name__ == "__main__":
    main()
