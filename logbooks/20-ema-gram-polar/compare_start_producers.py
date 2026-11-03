# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare selected declared input routes, preserving opaque boundary limits."""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

from compare_outer_bindings import normalized, rooted_differences


def canonical_trace(value):
    result = copy.deepcopy(value)
    for boundary in result["boundaries"]:
        if boundary["reason"] == "declared product frontier":
            row = result["records"][boundary["node"]]
            labels = re.findall(r'op_name="([^"]+)"', row["attributes"])
            assert len(labels) == 1
            label = labels[0]
            roles = [x for x in ("/gram_direction/", "/update_gram_state/") if x in label]
            assert len(roles) == 1
            boundary["role"] = label[label.index(roles[0]) :]
    return result


def by_group(graph):
    result = {}
    for start in graph["trace"]["starts"]:
        assert len(start["uses"]) == 1
        use = start["uses"][0]
        group = use["group"]
        assert group not in result
        result[group] = start
    assert set(result) == {"gate_proj", "down_proj", "up_proj", "o_proj"}
    return result


def compare(report):
    assert report["completed"]
    graphs = report["artifacts"]
    reference = by_group(graphs["matrix_untouched"])
    result = {}
    for name, graph in graphs.items():
        assert graph["sha256_verified"]
        result[name] = {}
        for group, selected in by_group(graph).items():
            value = canonical_trace(selected["declared_input_trace"])
            base = canonical_trace(reference[group]["declared_input_trace"])
            assert not value["missing_records"]
            root = value["records"][value["root"]]
            assert root["opcode"] == "all-gather"
            sig, _, _ = normalized(value, name)
            result[name][group] = dict(
                same_normalized_signature_as_untouched=sig == normalized(base, "matrix_untouched")[0],
                normalized_signature=sig,
                rooted_differences=rooted_differences(base, value, name),
                start=selected["start"],
                final_product_use=selected["uses"][0],
                declared_input=root,
                records=len(value["records"]),
                edges=len(value["edges"]),
                boundaries=[dict(**b, record=value["records"][b["node"]]) for b in value["boundaries"]],
            )
    assert all(v["same_normalized_signature_as_untouched"] for v in result["matrix_evaluate"].values())
    return dict(
        graphs=result,
        limitations=[
            "Selection uses output dependencies and a unique immediate Done frontier per group, not list positions.",
            "Auxiliary Start input routes do not make Start/Done/Concat outputs transparent.",
            "Matching bounded signatures is not semantic identity, runtime values, precision, reduction order or cause.",
            "Raw edges retain caller contexts and repeated positions; recursive signatures do not distinguish sharing.",
            "Both full28leaf numerical substitution gates remain failed.",
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
    args.output.write_text(json.dumps(compare(json.loads(raw)), indent=2) + "\n")


if __name__ == "__main__":
    main()
