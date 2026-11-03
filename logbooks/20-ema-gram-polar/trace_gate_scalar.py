# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Locate the gate trace-scalar route inside the previously extracted dependencies."""

import argparse
import hashlib
import json
from pathlib import Path


def route(value):
    targets = [
        i
        for i, row in enumerate(value["records"])
        if row.get("opcode") == "reduce" and "/relative_gram_power/jit(trace)/reduce_sum" in row["attributes"]
    ]
    assert len(targets) == 1, "Missing or ambiguous trace-scalar producer"
    current = targets[0]
    result, seen = [], set()
    while True:
        assert current not in seen
        seen.add(current)
        row = value["records"][current]
        result.append(dict(node_index=current, **row))
        if row["opcode"] == "broadcast":
            assert "/relative_gram_power/jit(_where)/broadcast_in_dim" in row["attributes"]
            break
        parents = [e for e in value["edges"] if e["target"] == current]
        assert len(parents) == 1, "Missing or ambiguous consumer in bounded trace"
        current = parents[0]["source"]
    ar = [r for r in result if r["opcode"] == "all-reduce"]
    assert len(ar) == 1 and ar[0]["tuple_path"] == [3]
    predicate = [r for r in result if r["opcode"] == "compare"]
    assert len(predicate) == 1
    scalar_operations = [r for r in result if r["opcode"] in ("multiply", "compare")]
    assert [r["opcode"] for r in scalar_operations] == ["multiply", "compare"]
    assert "direction=GT" in scalar_operations[1]["attributes"]
    side_inputs = []
    for operation, literal in zip(scalar_operations, ("0.0009765625", "0"), strict=True):
        edges = [e for e in value["edges"] if e["source"] == operation["node_index"] and e["operand_index"] == 1]
        assert len(edges) == 1
        broadcast = value["records"][edges[0]["target"]]
        assert broadcast["opcode"] == "broadcast"
        children = [e for e in value["edges"] if e["source"] == edges[0]["target"]]
        assert len(children) == 1
        constant = value["records"][children[0]["target"]]
        assert constant["opcode"] == "constant" and "constant(" + literal + ")" in constant["instruction"]
        side_inputs.append(dict(operation=operation["opcode"], broadcast=broadcast, constant=constant))
    return dict(
        producer_to_predicate_broadcast=result,
        side_input_literals=side_inputs,
        allreduce_selected_projection=3,
        copy_pairs=sum(r["opcode"] == "copy-start" for r in result),
        scope="One route in the gate numerator dependency trace, not all uses or runtime values.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.sha256
    r = json.loads(raw)
    assert r["completed"]
    result = dict(
        report_sha256=args.sha256,
        graphs={n: route(g["trace"]["gate_proj"]["normalization_numerator"]) for n, g in r["artifacts"].items()},
        limitations=(
            "Extra copy route also occurs in the passing paired-capture control. "
            "S(1) tags and channel IDs are recorded, not assigned numerical meaning. No causal claim."
        ),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
