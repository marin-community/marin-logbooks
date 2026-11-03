# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare upstream product and collective input routes, retaining opaque outputs."""

import argparse
import hashlib
import json
from pathlib import Path

from compare_outer_bindings import normalized
from compare_start_producers import canonical_trace
from trace_capture_initializer_concat import ROLE


def indexed(graph):
    result = {}
    targets = graph["trace"]["targets"]
    assert set(targets) == {ROLE}
    for label, target in targets.items():
        assert target["endpoint"]["opcode"] == "custom-call"
        assert 'custom_call_target="ConcatBitcast"' in target["endpoint"]["attributes"]
        entries = [(label, target["endpoint"], target["operands"], 4)]
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
    assert len(result) == 4
    return result


def rooted_differences(left, right, variant):
    _, lf, lc = normalized(left, "matrix_capture")
    _, rf, rc = normalized(right, variant)
    seen, differences = set(), []

    def visit(i, j, path):
        if (i, j) in seen:
            return
        seen.add((i, j))
        changed = [k for k in sorted(set(lf[i]) | set(rf[j])) if lf[i].get(k) != rf[j].get(k)]
        a, b = sorted(lc.get(i, [])), sorted(rc.get(j, []))
        edge_mismatch = [p for p, _ in a] != [p for p, _ in b]
        stop = lf[i]["opcode"] != rf[j]["opcode"] or edge_mismatch
        if changed or edge_mismatch:
            differences.append(
                dict(
                    operand_path=path,
                    left_node=i,
                    right_node=j,
                    changed_fields=changed,
                    left={k: lf[i].get(k) for k in changed},
                    right={k: rf[j].get(k) for k in changed},
                    operand_positions_differ=edge_mismatch,
                    unaligned_subtrees_not_compared=stop,
                )
            )
        if not stop:
            for (p, k), (_, l) in zip(a, b, strict=True):
                visit(k, l, [*path, p])

    visit(left["root"], right["root"], [])
    return differences


def compare(report):
    assert report["completed"]
    graphs = report["artifacts"]
    assert set(graphs) == {"matrix_capture", "matrix_substitute"}
    base = indexed(graphs["matrix_capture"])
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
                same_normalized_signature_as_capture=sig == normalized(reference, "matrix_capture")[0],
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
    assert all(v["same_normalized_signature_as_capture"] for v in result["matrix_capture"].values())
    return dict(
        graphs=result,
        limitations=[
            "New routes compare substitution against capture only; no untouched/evaluator counterpart is invented.",
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
