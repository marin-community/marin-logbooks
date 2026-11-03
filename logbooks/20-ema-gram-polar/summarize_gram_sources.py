# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve source frontiers and scoped differences from the saved Gram-source report."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from compare_gram_source_inputs import compare, indexed
from consolidate_compiler_records import identity


def tuple_leaves(shape):
    assert shape.startswith("(") and shape.endswith(")")
    depth, start, result = 0, 1, []
    for i, char in enumerate(shape[1:-1], 1):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            result.append(shape[start:i].strip())
            start = i + 1
    result.append(shape[start:-1].strip())
    return [re.sub(r"/\*index=\d+\*/", "", v) for v in result]


def summarize(report):
    comparisons = compare(report)
    result = {}
    for name, graph in report["artifacts"].items():
        values = indexed(graph)
        sources = {}
        for label, value in values.items():
            if not label.startswith("down_proj/source/"):
                continue
            sources[label] = [value["records"][b["node"]] for b in value["boundaries"]]
        gradients = [sources[f"down_proj/source/refresh_product/{i}"] for i in (0, 1)]
        assert all(len(rows) == 1 and rows[0]["opcode"] == "parameter" for rows in gradients)
        assert identity(gradients[0][0]) == identity(gradients[1][0])
        assert re.search(r'(gradients\[6\]|bank\[0\]\[6\]|bank\[0\]\[0\]\[6\])"', gradients[0][0]["attributes"])
        changes = {}
        for label, comparison in comparisons["graphs"][name].items():
            if comparison["same_normalized_signature_as_untouched"]:
                continue
            value = values[label]
            base = indexed(report["artifacts"]["matrix_untouched"])[label]
            changes[label] = [
                dict(
                    comparison=d,
                    original_record=base["records"][d["left_node"]],
                    changed_record=value["records"][d["right_node"]],
                )
                for d in comparison["rooted_differences"]
            ]
        # Passing paired capture has only sibling tuple-tag differences on this route.
        if name == "matrix_capture":
            assert list(changes) == ["down_proj/source/direct_product/0"]
            assert len(next(iter(changes.values()))) == 1
            change = next(iter(changes.values()))[0]
            left, right = change["original_record"], change["changed_record"]
            assert change["comparison"]["changed_fields"] == ["shape"]
            assert left["opcode"] == right["opcode"] == "all-reduce"
            assert left["tuple_path"] == right["tuple_path"] == [5]
            a, b = tuple_leaves(left["shape"]), tuple_leaves(right["shape"])
            assert len(a) == len(b) == 6 and a[5] == b[5]
            assert [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y] == [3, 4]
            assert left["shape"].replace("S(1)", "") == right["shape"].replace("S(1)", "")
        up_views = []
        for label, value in values.items():
            if label.startswith("up_proj/opaque/"):
                matches = [r for r in value["records"] if r["opcode"] == "all-reduce"]
                assert len(matches) == 1
                up_views.append(matches[0])
        assert len(up_views) == 4 and len({identity(r) for r in up_views}) == 1
        assert all(r["tuple_path"] == ([5] if name == "matrix_substitute" else [4]) for r in up_views)
        result[name] = dict(
            source_frontiers=sources,
            refresh_operands_reach_same_declared_parameter=True,
            changed_routes=changes,
            shared_up_normalization_projection=up_views[0],
        )
    return dict(
        graphs=result,
        limitations=[
            "Same declared parameter identity within a graph does not establish runtime identity across graphs.",
            "S(1) and channel IDs are compiler declarations, without inferred runtime precision or order.",
            "Four up views share one record; up remains a passing numerical control.",
            "Difference traversal stops at unaligned subtrees; their remaining equality is unestablished.",
            "All original custom-call and while outputs remain opaque; both numerical gates remain failed.",
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
