# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect declared inputs of opaque endpoints without assigning their semantics."""

import re

from compiler_operation_inventory import CALLS
from trace_capture_normalization import parse_nodes
from trace_outer_bindings import trace

TARGET = re.compile(r'custom_call_target="([^"]+)"')
SELECTED = {"ConcatBitcast", "AsyncCollectiveDone"}


def build_scopes(graph, outer, extra):
    bodies = {**graph["producer_bodies"], **graph.get("first_gram_path_bodies", {}), **extra}
    scopes = {s: parse_nodes(lines) for s, lines in bodies.items()}
    sparse = {}
    for roles in outer.values():
        for value in roles.values():
            for row in value["records"]:
                scope = row["computation"]
                nodes, root = sparse.setdefault(scope, ({}, None))
                if row["name"] in nodes:
                    assert nodes[row["name"]]["instruction"] == row["instruction"]
                nodes[row["name"]] = {k: v for k, v in row.items() if k not in ("computation", "context", "tuple_path")}
                if row["instruction"].startswith("ROOT "):
                    assert root in (None, row["name"])
                    sparse[scope] = nodes, row["name"]
    # Complete saved bodies take precedence over the sparse traced subset.
    for scope, (nodes, root) in sparse.items():
        if scope not in scopes:
            scopes[scope] = nodes, root
    entries = graph["lineage_diagnostics"]["entry_instructions"]
    entry = {r["computation"] for r in graph["fusion_callers"] if any("%" + r["name"] + " =" in l for l in entries)}
    assert len(entry) == 1
    scopes[entry.pop()] = parse_nodes(entries)
    return scopes


def caller_bindings(scopes, context, target_scope):
    bindings, prefix, callee = None, (), None
    for identity in context:
        scope, name = identity.split("/", 1)
        if callee is not None:
            assert scope == callee
        caller = scopes[scope][0][name]
        assert caller["opcode"] == "fusion"
        callee = CALLS.search(caller["attributes"])[1]
        bindings = [(scope, n, bindings, prefix) for n in caller["operands"]]
        prefix = (*prefix, identity)
    if context:
        assert callee == target_scope
    return bindings


def analyze(graph, outer, extra=None):
    scopes = build_scopes(graph, outer, extra or {})
    polar = {
        row["operation"]: f"{g}/{row['iteration']}/{row['product']}"
        for g, chain in graph["polar_lineage"]["output_groups"].items()
        for row in chain
    }
    endpoints = {}
    for group, roles in outer.items():
        for role, value in roles.items():
            for boundary in value["boundaries"]:
                row = value["records"][boundary["node"]]
                match = TARGET.search(row.get("attributes", ""))
                if row.get("opcode") != "custom-call" or match is None or match[1] not in SELECTED:
                    continue
                key = (tuple(row["context"]), row["computation"], row["name"])
                if key not in endpoints:
                    bindings = caller_bindings(scopes, key[0], key[1])
                    # A sparse body lacking its root is an explicit missing body.
                    usable = {k: v for k, v in scopes.items() if v[1] is not None}
                    endpoints[key] = dict(
                        endpoint=row,
                        custom_call_target=match[1],
                        root_uses=[],
                        operand_traces=[
                            dict(operand_position=i, trace=trace(usable, key[1], n, bindings, key[0], polar))
                            for i, n in enumerate(row["operands"])
                        ],
                    )
                endpoints[key]["root_uses"].append(dict(group=group, role=role))
    return dict(
        endpoints=list(endpoints.values()),
        limitations=(
            "Auxiliary declared operand dependencies only. Original custom-call outputs remain opaque. "
            "Reaching AsyncCollectiveStart records does not establish exact start/done semantics or runtime values."
        ),
    )
