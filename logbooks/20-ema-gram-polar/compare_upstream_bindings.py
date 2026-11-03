# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare upstream product and collective input routes, retaining opaque outputs."""

import argparse
import hashlib
import json
from pathlib import Path

from compare_outer_bindings import normalized, rooted_differences
from compare_start_producers import canonical_trace
from trace_frontier_collectives import common_start


def indexed(graph):
    result = {}
    targets = graph["trace"]["targets"]
    assert len(targets) == 8
    for label, target in targets.items():
        count = {"X_Xt": 2, "B_X": 2, "ConcatBitcast": 4, "AsyncCollectiveDone": 10}[label.split("/")[-1]]
        entries = [(label, target["endpoint"], target["operands"], count)]
        if label.endswith("/AsyncCollectiveDone"):
            start, positions = common_start(target["operands"])
            assert target["matched_start"]["record"] == start
            assert target["matched_start"]["positions"] == positions
            entries.append((label + "/Start", start, target["matched_start"]["operands"], 1))
        for key, endpoint, operands, n in entries:
            assert len(endpoint["operands"]) == n
            assert sorted(v["operand_position"] for v in operands) == list(range(n))
            for operand in operands:
                position = operand["operand_position"]
                value = operand["trace"]
                assert not value["missing_records"]
                root = value["records"][value["root"]]
                assert root["name"] == endpoint["operands"][position]
                assert root["context"] == endpoint["context"]
                assert root["computation"] == endpoint["computation"]
                assert not root["tuple_path"]
                result[f"{key}/{position}"] = value
    assert len(result) == 38
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
            "Endpoints selected by saved dependencies, caller contexts and mapped polar roles.",
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
