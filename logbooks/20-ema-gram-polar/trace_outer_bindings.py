# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trace output and normalization bindings, retaining opaque and polar boundaries."""

import hashlib
import json
import re

from compiler_operation_inventory import CALLS, TUPLE_INDEX
from trace_capture_normalization import parse_nodes

GROUPS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
TARGETS = ("gate_proj", "down_proj", "up_proj", "o_proj")
OPAQUE = {
    "custom-call",
    "while",
    "call",
    "conditional",
    "async-start",
    "async-done",
    "async-collective-start",
    "async-collective-done",
}


def trace(scopes, scope, name, bindings, context, polar):
    records, edges, missing, boundaries, memo = [], [], set(), [], {}

    def walk(scope, name, bindings, context, path=()):
        key = (context, scope, name, path)
        if key in memo:
            return memo[key]
        index = len(records)
        memo[key] = index
        if scope not in scopes or name not in scopes[scope][0]:
            missing.add(scope + "/" + name)
            records.append(dict(computation=scope, name=name, context=context, tuple_path=path, missing=True))
            return index
        node = scopes[scope][0][name]
        records.append(dict(computation=scope, context=context, tuple_path=path, **node))
        op, operands = node["opcode"], node["operands"]
        identity = scope + "/" + name

        def follow(position, s=scope, n=None, b=bindings, c=context, p=()):
            child = walk(s, operands[position] if n is None else n, b, c, p)
            edges.append(dict(source=index, operand_index=position, target=child))

        if identity in polar:
            boundaries.append(dict(node=index, reason="polar product anchor", role=polar[identity]))
        elif op == "parameter":
            if bindings is None:
                boundaries.append(dict(node=index, reason="entry parameter"))
            else:
                s, n, b, c = bindings[node["parameter_index"]]
                follow(node["parameter_index"], s, n, b, c, path)
        elif op == "get-tuple-element":
            follow(0, p=(int(TUPLE_INDEX.search(node["attributes"])[1]), *path))
        elif op == "tuple":
            assert path
            follow(path[0], p=path[1:])
        elif op == "fusion":
            callee = CALLS.search(node["attributes"])[1]
            if callee not in scopes:
                missing.add(callee)
            else:
                args = [(scope, n, bindings, context) for n in operands]
                follow(0, callee, scopes[callee][1], args, (*context, identity), path)
        elif op == "all-reduce" and len(operands) > 1:
            assert path
            follow(path[0], p=path[1:])
        elif op == "copy-done":
            follow(0, p=(0, *path))
        elif op == "copy-start":
            assert path and path[0] in (0, 1)
            follow(0, p=path[1:])
        elif op == "slice-done":
            assert not path and scopes[scope][0][operands[0]]["opcode"] == "slice-start"
            follow(0)
        elif op == "slice-start":
            assert not path and len(operands) == 1
            follow(0)
        elif op in OPAQUE or path:
            boundaries.append(dict(node=index, reason="opaque operation or unsupported tuple projection"))
        else:
            for i in range(len(operands)):
                follow(i)
        return index

    root = walk(scope, name, bindings, context)
    children = {}
    for edge in edges:
        children.setdefault(edge["source"], []).append((edge["operand_index"], edge["target"]))
    labels = {b["node"]: b for b in boundaries}
    hashes = {}

    def signature(index):
        if index in hashes:
            return hashes[index]
        row = records[index]
        attributes = re.sub(r",? metadata=\{[^}]*\}", "", row.get("attributes", ""))
        attributes = re.sub(r"%[A-Za-z0-9_.-]+", "<identifier>", attributes)
        fields = dict(
            shape=row.get("shape"),
            opcode=row.get("opcode"),
            attributes=attributes,
            tuple_path=row["tuple_path"],
            missing=row.get("missing", False),
        )
        if index in labels:
            fields["boundary"] = {k: v for k, v in labels[index].items() if k != "node"}
            if labels[index]["reason"] == "entry parameter":
                fields["source_label"] = re.findall(r'op_name="([^"]+)"', row["attributes"])
                fields["parameter_index"] = row["parameter_index"]
        fields["ordered_edges"] = [(i, signature(j)) for i, j in sorted(children.get(index, []))]
        hashes[index] = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
        return hashes[index]

    return dict(
        root=root,
        records=records,
        edges=edges,
        missing_records=sorted(missing),
        boundaries=boundaries,
        rooted_signature=signature(root),
        signature_scope=(
            "Ordered rooted declarations/connections to boundaries; identifiers/metadata removed except entry labels. "
            "Not runtime values or complete semantic identity; sharing is retained in raw edges "
            "but not distinguished by the recursive signature."
        ),
    )


def analyze_graph(graph, extra=None):
    bodies = {**graph["producer_bodies"], **graph.get("first_gram_path_bodies", {}), **(extra or {})}
    scopes = {s: parse_nodes(lines) for s, lines in bodies.items()}
    entries = graph["lineage_diagnostics"]["entry_instructions"]
    entry_names = {
        r["computation"] for r in graph["fusion_callers"] if any("%" + r["name"] + " =" in l for l in entries)
    }
    assert len(entry_names) == 1
    entry = entry_names.pop()
    scopes[entry] = parse_nodes(entries)
    nodes, root = scopes[entry]
    polar = {
        row["operation"]: f"{g}/{row['iteration']}/{row['product']}"
        for g, chain in graph["polar_lineage"]["output_groups"].items()
        for row in chain
    }
    result = {}
    for group in TARGETS:
        divisions = graph["normalization_by_group"][group]
        assert len(divisions) == 1
        scope, name = divisions[0].split("/")
        callers = [r for r in nodes.values() if r["opcode"] == "fusion" and CALLS.search(r["attributes"])[1] == scope]
        assert len(callers) == 1
        caller = callers[0]
        args = [(entry, n, None, ()) for n in caller["operands"]]
        division = scopes[scope][0][name]
        assert division["opcode"] == "divide" and len(division["operands"]) == 2
        result[group] = {
            role: trace(scopes, scope, value, args, (entry + "/" + caller["name"],), polar)
            for role, value in zip(
                ("normalization_numerator", "normalization_denominator"), division["operands"], strict=True
            )
        }
        result[group]["final_direction"] = trace(
            scopes, entry, nodes[root]["operands"][GROUPS.index(group)], None, (), polar
        )
    return result
