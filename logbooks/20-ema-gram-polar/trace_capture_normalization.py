# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trace first-product operands to normalization boundaries in captured HLO records."""

import argparse
import hashlib
import json
from pathlib import Path

from compiler_operation_inventory import CALLS, INSTRUCTION, OPERAND, TUPLE_INDEX


def parse_nodes(lines):
    nodes, root = {}, None
    for line in lines:
        match = INSTRUCTION.match(line)
        if match is None:
            continue
        name, shape, opcode, arguments, attributes = match.groups()
        nodes[name] = dict(
            name=name,
            shape=shape,
            opcode=opcode,
            operands=OPERAND.findall(arguments),
            parameter_index=int(arguments) if opcode == "parameter" else None,
            attributes=attributes,
            instruction=line.strip(),
        )
        if line.lstrip().startswith("ROOT "):
            root = name
    assert root is not None
    return nodes, root


def trace(
    scopes, start_scope, start_name, bindings, context=(), follow_concat_operands=False, inspect_async_allgather=False
):
    records, found, missing, boundaries, visited = [], [], set(), [], set()
    auxiliary = []

    def walk(scope, name, arguments, call_context, path=()):
        key = (call_context, scope, name, path)
        if key in visited:
            return
        visited.add(key)
        if scope not in scopes or name not in scopes[scope][0]:
            missing.add(scope + "/" + name)
            return
        node = scopes[scope][0][name]
        row = dict(computation=scope, context=list(call_context), selected_tuple_path=list(path), **node)
        records.append(row)
        op, operands = node["opcode"], node["operands"]
        if op == "divide" and "/gram_direction/vmap()/div" in node["attributes"]:
            assert not path
            found.append(row)
            return
        if op == "parameter":
            if arguments is None:
                boundaries.append(dict(reason="entry parameter", **row))
                return
            index = node["parameter_index"]
            assert index < len(arguments)
            walk(*arguments[index], path)
            return
        if op == "get-tuple-element":
            walk(scope, operands[0], arguments, call_context, (int(TUPLE_INDEX.search(node["attributes"])[1]), *path))
            return
        if op == "tuple":
            assert path
            walk(scope, operands[path[0]], arguments, call_context, path[1:])
            return
        if op == "slice-done":
            assert not path and len(operands) == 1
            assert scopes[scope][0][operands[0]]["opcode"] == "slice-start"
            walk(scope, operands[0], arguments, call_context)
            return
        if op == "slice-start":
            assert not path and len(operands) == 1
            walk(scope, operands[0], arguments, call_context)
            return
        if op == "copy-done":
            walk(scope, operands[0], arguments, call_context, (0, *path))
            return
        if op == "copy-start":
            assert path and path[0] in (0, 1) and len(operands) == 1
            walk(scope, operands[0], arguments, call_context, path[1:])
            return
        if op == "fusion":
            callee = CALLS.search(node["attributes"])[1]
            if callee not in scopes:
                missing.add(callee)
                return
            child_bindings = [(scope, value, arguments, call_context) for value in operands]
            walk(callee, scopes[callee][1], child_bindings, (*call_context, scope + "/" + name), path)
            return
        if op == "all-reduce" and len(operands) > 1:
            assert path
            walk(scope, operands[path[0]], arguments, call_context, path[1:])
            return
        if op in (
            "custom-call",
            "while",
            "call",
            "conditional",
            "async-start",
            "async-done",
            "async-collective-start",
            "async-collective-done",
        ):
            conservative = (
                follow_concat_operands
                and op == "custom-call"
                and 'custom_call_target="ConcatBitcast"' in node["attributes"]
            )
            boundaries.append(dict(reason="opaque operation", declared_operands_followed=conservative, **row))
            if inspect_async_allgather and 'custom_call_target="AsyncCollectiveDone"' in node["attributes"]:
                for index, operand in enumerate(operands):
                    if scopes[scope][0][operand]["opcode"] == "all-gather":
                        auxiliary.append(
                            dict(
                                opaque_call=row,
                                operand_index=index,
                                interpretation="Declared all-gather operand only; opaque result semantics unresolved",
                                trace=trace(
                                    scopes,
                                    scope,
                                    operand,
                                    arguments,
                                    (*call_context, f"{scope}/{name}:allgather_operand{index}"),
                                    follow_concat_operands,
                                    False,
                                ),
                            )
                        )
            if conservative:
                assert not path
                for index, operand in enumerate(operands):
                    walk(scope, operand, arguments, (*call_context, f"{scope}/{name}:operand{index}"))
            return
        assert not path, (scope, name, path)
        for operand in operands:
            walk(scope, operand, arguments, call_context)

    walk(start_scope, start_name, bindings, context)
    return dict(
        normalization_divisions=found,
        opaque_allgather_branches=auxiliary,
        missing_records=sorted(missing),
        unresolved_boundaries=boundaries,
        records=records,
    )


def analyze(source, supplements=(), follow_concat_operands=False, inspect_async_allgather=False):
    graphs = {}
    for name, graph in source["artifacts"].items():
        bodies = dict(graph["producer_bodies"])
        for supplement in supplements:
            extra = supplement["artifacts"][name]
            assert extra["sha256_verified"]
            assert extra["input_sha256"] == source["manifest"]["artifacts"][name]["sha256"]
            bodies.update(extra["bodies"])
        scopes = {scope: parse_nodes(body) for scope, body in bodies.items()}
        # Identify entry from the captured parent records, never an arbitrary first caller.
        entries = graph["lineage_diagnostics"]["entry_instructions"]
        entry_names = {INSTRUCTION.match(line)[1] for line in entries if INSTRUCTION.match(line)}
        parents = {row["calls"]: row for row in graph["fusion_callers"] if row["name"] in entry_names}
        entry_scopes = {row["computation"] for row in parents.values()}
        assert len(entry_scopes) == 1
        entry = entry_scopes.pop()
        scopes[entry] = parse_nodes(entries)
        operations = {row["computation"] + "/" + row["name"]: row for row in graph["selected_operations"]}
        groups = {}
        for group, chain in graph["polar_lineage"]["output_groups"].items():
            product = operations[chain[0]["operation"]]
            candidates = [
                row
                for row in graph["fusion_callers"]
                if row["calls"] == product["computation"] and row["name"] in entry_names
            ]
            assert len(candidates) == 1
            parent = candidates[0]
            bindings = [(entry, value, None, ()) for value in parent["operands"]]
            groups[group] = [
                dict(
                    operand_index=index,
                    first_product=chain[0]["operation"],
                    **trace(
                        scopes,
                        product["computation"],
                        operand,
                        bindings,
                        (entry + "/" + parent["name"],),
                        follow_concat_operands,
                        inspect_async_allgather,
                    ),
                )
                for index, operand in enumerate(product["operands"])
            ]
        graphs[name] = groups
    return dict(
        graphs=graphs,
        limitations=(
            "Static parameter and tuple-leaf bindings, stopping at normalization divisions "
            "or explicit opaque boundaries. Optional ConcatBitcast traversal follows declared operands conservatively; "
            "Repeated product operands retain separate positions. Reaching a division is not a runtime-value identity. "
            "Both numerical capture gates failed; no original-path or causal claim."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, action="append", default=[])
    parser.add_argument("--follow-concat-operands", action="store_true")
    parser.add_argument("--inspect-async-allgather", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    source = json.loads(raw)
    assert source["completed"]
    supplements = [json.loads(path.read_bytes()) for path in args.supplement]
    for supplement in supplements:
        assert supplement["completed"]
    result = analyze(source, supplements, args.follow_concat_operands, args.inspect_async_allgather)
    result["input_report_sha256"] = hashlib.sha256(raw).hexdigest()
    result["supplement_sha256"] = [hashlib.sha256(path.read_bytes()).hexdigest() for path in args.supplement]
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
