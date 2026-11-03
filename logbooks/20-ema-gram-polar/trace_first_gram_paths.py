# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trace selected second-product operand paths back to first-Gram products."""

import argparse
import hashlib
import json
from pathlib import Path

from compiler_operation_inventory import CALLS, INSTRUCTION, TUPLE_INDEX
from trace_capture_normalization import parse_nodes


def trace(scopes, scope, name, bindings, context, targets):
    records, edges, missing, boundaries, reached = {}, [], set(), [], set()

    def walk(scope, name, bindings, context, path=()):
        key = json.dumps([context, scope, name, path])
        if key in records:
            return key
        if scope not in scopes or name not in scopes[scope][0]:
            missing.add(scope + "/" + name)
            return key
        node = scopes[scope][0][name]
        records[key] = dict(context=list(context), computation=scope, tuple_path=list(path), **node)
        identity = scope + "/" + name
        if identity in targets:
            assert not path
            reached.add(identity)
            return key
        op, operands = node["opcode"], node["operands"]

        def follow(index, dest_scope=scope, dest_bindings=bindings, dest_context=context, dest_name=None, dest_path=()):
            child = walk(
                dest_scope, operands[index] if dest_name is None else dest_name, dest_bindings, dest_context, dest_path
            )
            edges.append(dict(source=key, operand_index=index, target=child))

        if op == "parameter":
            if bindings is None:
                boundaries.append(dict(reason="entry parameter", identity=key))
            else:
                s, n, b, c = bindings[node["parameter_index"]]
                follow(node["parameter_index"], s, b, c, n, path)
        elif op == "get-tuple-element":
            follow(0, dest_path=(int(TUPLE_INDEX.search(node["attributes"])[1]), *path))
        elif op == "tuple":
            assert path
            follow(path[0], dest_path=path[1:])
        elif op == "fusion":
            callee = CALLS.search(node["attributes"])[1]
            if callee not in scopes:
                missing.add(callee)
            else:
                arguments = [(scope, n, bindings, context) for n in operands]
                follow(0, callee, arguments, (*context, identity), scopes[callee][1], path)
        elif op == "all-reduce" and len(operands) > 1:
            assert path
            follow(path[0], dest_path=path[1:])
        elif op == "copy-done":
            follow(0, dest_path=(0, *path))
        elif op == "copy-start":
            assert path and path[0] in (0, 1)
            follow(0, dest_path=path[1:])
        elif op in ("custom-call", "while", "call", "conditional", "async-start", "async-done") or path:
            boundaries.append(dict(reason="opaque operation or unsupported tuple projection", identity=key))
        else:
            for index in range(len(operands)):
                follow(index)
        return key

    root = walk(scope, name, bindings, context)
    identifiers = {key: index for index, key in enumerate(records)}
    compact = {identifiers[key]: {k: v for k, v in row.items() if k != "instruction"} for key, row in records.items()}
    for edge in edges:
        edge["source"] = identifiers[edge["source"]]
        edge["target"] = identifiers.get(edge["target"], edge["target"])
    for boundary in boundaries:
        boundary["identity"] = identifiers[boundary["identity"]]
    return dict(
        root=identifiers.get(root, root),
        records=compact,
        edges=edges,
        missing_records=sorted(missing),
        boundaries=boundaries,
        reached_first_products=sorted(reached),
        collectives=[k for k, v in compact.items() if v["opcode"] in ("all-reduce", "reduce-scatter", "all-gather")],
    )


def analyze(source, supplement):
    graphs = {}
    for name, graph in source["artifacts"].items():
        extra = supplement["artifacts"][name]
        assert extra["sha256_verified"] and extra["input_sha256"] == source["manifest"]["artifacts"][name]["sha256"]
        bodies = {**graph["producer_bodies"], **extra["bodies"]}
        scopes = {s: parse_nodes(lines) for s, lines in bodies.items()}
        entries = graph["lineage_diagnostics"]["entry_instructions"]
        entry_names = {INSTRUCTION.match(l)[1] for l in entries if INSTRUCTION.match(l)}
        parents = [r for r in graph["fusion_callers"] if r["name"] in entry_names]
        entry_scopes = {r["computation"] for r in parents}
        assert len(entry_scopes) == 1
        entry = entry_scopes.pop()
        scopes[entry] = parse_nodes(entries)
        groups = {}
        for group in ("q_proj", "k_proj", "v_proj"):
            chain = graph["polar_lineage"]["output_groups"][group]
            targets = {
                graph["polar_lineage"]["output_groups"][g][0]["operation"] for g in ("q_proj", "k_proj", "v_proj")
            }
            first, second = chain[:2]
            scope, node_name = second["operation"].split("/")
            callers = [r for r in parents if r["calls"] == scope]
            assert len(callers) == 1
            parent = callers[0]
            bindings = [(entry, n, None, ()) for n in parent["operands"]]
            node = scopes[scope][0][node_name]
            assert len(node["operands"]) == 2
            operands = []
            for i, n in enumerate(node["operands"]):
                path = trace(scopes, scope, n, bindings, (entry + "/" + parent["name"],), targets)
                assert not path["missing_records"], path["missing_records"]
                assert path["reached_first_products"] == [first["operation"]], (
                    name,
                    group,
                    path["reached_first_products"],
                )
                first_scope = first["operation"].split("/")[0]
                projections = [
                    r["tuple_path"]
                    for r in path["records"].values()
                    if r["computation"] == first_scope and r["name"] == scopes[first_scope][1]
                ]
                assert projections == [[0]], projections
                path["first_product_fusion_output_path"] = [0]
                operands.append(dict(operand_index=i, **path))
            groups[group] = dict(
                first_product=first,
                second_product=second,
                second_product_caller={k: parent[k] for k in ("computation", "name", "calls", "operands", "shape")},
                operands=operands,
            )
        graphs[name] = groups
    return dict(
        graphs=graphs,
        limitations=(
            "Static declared paths from first to second polar product; explicit opaque boundaries remain. "
            "No runtime precision, summation order, numerical identity or cause. "
            "Both numerical substitution reference gates failed."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--supplement-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = []
    for path, sha in ((args.inventory, args.inventory_sha256), (args.supplement, args.supplement_sha256)):
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == sha
        obj = json.loads(raw)
        assert obj["completed"]
        inputs.append(obj)
    result = analyze(*inputs)
    result["input_hashes"] = [args.inventory_sha256, args.supplement_sha256]
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
