# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare dependency-selected product operand routes, retaining opaque boundaries."""

import argparse
import hashlib
import json
from pathlib import Path

from compare_outer_bindings import normalized, rooted_differences
from compare_start_producers import canonical_trace


def indexed(graph):
    targets = graph["trace"]["targets"]
    assert len(targets) == 5
    result = {}
    for label, target in targets.items():
        assert target["endpoint"]["opcode"] in ("convolution", "dot")
        assert len(target["endpoint"]["operands"]) == 2
        assert sorted(v["operand_position"] for v in target["operands"]) == [0, 1]
        for operand in target["operands"]:
            position = operand["operand_position"]
            value = operand["trace"]
            assert not value["missing_records"]
            root = value["records"][value["root"]]
            assert root["name"] == target["endpoint"]["operands"][position]
            assert root["context"] == target["endpoint"]["context"]
            assert root["computation"] == target["endpoint"]["computation"]
            result[f"{label}/{position}"] = value
    assert len(result) == 10
    return result


def compare(report):
    assert report["completed"]
    graphs = report["artifacts"]
    base = indexed(graphs["matrix_untouched"])
    result = {}
    for name, graph in graphs.items():
        assert graph["sha256_verified"]
        values = indexed(graph)
        assert values.keys() == base.keys()
        result[name] = {}
        for key, original in values.items():
            value, reference = canonical_trace(original), canonical_trace(base[key])
            sig, _, _ = normalized(value, name)
            result[name][key] = dict(
                normalized_signature=sig,
                same_normalized_signature_as_untouched=sig == normalized(reference, "matrix_untouched")[0],
                records=len(value["records"]),
                edges=len(value["edges"]),
                rooted_differences=rooted_differences(reference, value, name),
                boundaries=[
                    dict(
                        **b,
                        record={
                            k: v
                            for k, v in value["records"][b["node"]].items()
                            if k in ("context", "computation", "name", "shape", "opcode", "tuple_path")
                        },
                    )
                    for b in value["boundaries"]
                ],
            )
    assert all(v["same_normalized_signature_as_untouched"] for v in result["matrix_evaluate"].values())
    return dict(
        graphs=result,
        limitations=[
            "Roles selected by saved output dependencies/source scopes, polar chain map, exact caller identities.",
            "Signatures retain declarations, attributes, constant literals, ordered connections and boundary labels.",
            "Instruction/computation identifiers and source wrappers normalized; channel/chain IDs retained.",
            "Difference walker stops at unaligned subtrees. Recursive hashes do not distinguish sharing.",
            "Opaque outputs remain opaque. No runtime values, precision, order, semantic identity or cause claim.",
            "Both full28leaf numerical substitution gates remain failed; no numerical intervention selected.",
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
