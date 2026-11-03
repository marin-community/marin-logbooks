# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trace declared start inputs selected at final-product operand frontiers."""

import re
from collections import defaultdict

from trace_boundary_inputs import build_scopes, caller_bindings
from trace_outer_bindings import trace


def identity(row):
    return tuple(row["context"]), row["computation"], row["name"]


def direct_product(row):
    return row.get("opcode") == "convolution" and bool(
        re.search(r'op_name="[^"]*/gram_direction/dot_general"', row.get("attributes", ""))
    )


def final_frontiers(outer):
    result = []
    for group, roles in outer.items():
        value = roles["final_direction"]
        records = value["records"]
        edges = defaultdict(list)
        for edge in value["edges"]:
            edges[edge["source"]].append(edge)
        boundaries = {b["node"]: b["reason"] for b in value["boundaries"]}
        # Every selected product is already in the rooted final-output dependency graph.
        for index, row in enumerate(records):
            if not direct_product(row):
                continue
            operands = sorted(edges[index], key=lambda e: e["operand_index"])
            assert [e["operand_index"] for e in operands] == [0, 1]
            for edge in operands:
                seen, endpoints = set(), []

                pending = [edge["target"]]
                while pending:
                    i = pending.pop()
                    if i in seen:
                        continue
                    seen.add(i)
                    node = records[i]
                    if i in boundaries or node.get("opcode") in ("convolution", "dot"):
                        endpoints.append(dict(record=node, reason=boundaries.get(i, "declared product frontier")))
                        continue
                    assert not node.get("missing")
                    pending.extend(e["target"] for e in sorted(edges[i], key=lambda e: e["operand_index"], reverse=True))

                result.append(
                    dict(group=group, product=row, operand_position=edge["operand_index"], endpoints=endpoints)
                )
    return result


def analyze(graph, outer, boundary, extra=None):
    combined = dict(outer)
    combined["auxiliary"] = {
        f"{i}/{j}": operand["trace"]
        for i, endpoint in enumerate(boundary["endpoints"])
        for j, operand in enumerate(endpoint["operand_traces"])
    }
    scopes = build_scopes(graph, combined, extra or {})
    usable = {k: v for k, v in scopes.items() if v[1] is not None}
    products = {
        scope + "/" + name: row.get("attributes", "")
        for scope, (nodes, _) in scopes.items()
        for name, row in nodes.items()
        if row.get("opcode") in ("convolution", "dot")
    }
    endpoints = {identity(e["endpoint"]): e for e in boundary["endpoints"]}
    frontiers = final_frontiers(outer)
    selected = {}
    for frontier in frontiers:
        for item in frontier["endpoints"]:
            row = item["record"]
            if 'custom_call_target="AsyncCollectiveDone"' not in row.get("attributes", ""):
                continue
            endpoint = endpoints[identity(row)]
            starts = {}
            for operand in endpoint["operand_traces"]:
                value = operand["trace"]
                matches = [
                    value["records"][b["node"]]
                    for b in value["boundaries"]
                    if 'custom_call_target="AsyncCollectiveStart"' in value["records"][b["node"]].get("attributes", "")
                ]
                assert len(matches) == 1
                starts[identity(matches[0])] = matches[0]
            assert len(endpoint["operand_traces"]) == 10 and len(starts) == 1
            key, start = next(iter(starts.items()))
            assert len(start["operands"]) == 1
            if key not in selected:
                bindings = caller_bindings(scopes, key[0], key[1])
                value = trace(usable, key[1], start["operands"][0], bindings, key[0], products)
                for label in value["boundaries"]:
                    if label["reason"] == "polar product anchor":
                        label["reason"] = "declared product frontier"
                # Raw tracer hashes predate the relabeled boundary; never use them for comparisons.
                value.pop("rooted_signature")
                value.pop("signature_scope")
                selected[key] = dict(start=start, uses=[], declared_input_trace=value)
            selected[key]["uses"].append(
                dict(
                    group=frontier["group"],
                    product=frontier["product"]["name"],
                    operand_position=frontier["operand_position"],
                    done=row,
                )
            )
    return dict(
        final_product_frontiers=frontiers,
        starts=list(selected.values()),
        limitations=(
            "Declared inputs only. Start/Done/Concat outputs remain opaque; "
            "product records are not runtime values or causal evidence."
        ),
    )
