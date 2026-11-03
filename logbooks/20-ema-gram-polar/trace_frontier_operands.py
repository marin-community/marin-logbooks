# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trace exact saved product-frontier operands without crossing opaque outputs."""

import json

from consolidate_compiler_records import declaration, identity, traces
from trace_boundary_inputs import build_scopes, caller_bindings
from trace_outer_bindings import trace


def select_targets(graph, reports, coverage):
    records = {}
    for report in reports.values():
        for _, value in traces(report):
            for row in value["records"]:
                key = identity(row)
                if key in records:
                    assert declaration(records[key]) == declaration(row)
                records[key] = row
    polar = {
        row["operation"]: dict(group=group, iteration=row["iteration"], product=row["product"])
        for group, chain in graph["polar_lineage"]["output_groups"].items()
        for row in chain
    }
    result = {}
    for frontier in coverage["frontier_instructions"]:
        if frontier["category"] != "product_inputs_not_fully_inspected":
            continue
        context, scope, name = frontier["identity"]
        assert frontier["projections"] == [[]]
        row = records[json.dumps([context, scope, name, []], sort_keys=True, separators=(",", ":"))]
        assert row["opcode"] in ("dot", "convolution") and len(row["operands"]) == 2
        role = polar.get(scope + "/" + name)
        if role is not None:
            assert frontier["polar_role"] == role
            label = f"{role['group']}/polar/{role['iteration']}/{role['product']}"
        else:
            assert frontier["polar_role"] is None
            groups = []
            for group, kinds in reports["product-inputs"]["groups"].items():
                for operand in kinds["product"]["operands"]:
                    if operand["operand_position"] == 0:
                        value = operand["trace"]
                        if any(identity(value["records"][b["node"]]) == identity(row) for b in value["boundaries"]):
                            groups.append(group)
            assert len(groups) == 1 and groups[0] in ("gate_proj", "up_proj")
            attrs = row["attributes"]
            phases = [p for p in ("relative_gram_power", "update_gram_state") if "/" + p + "/" in attrs]
            assert len(phases) == 1
            label = f"{groups[0]}/{phases[0]}"
        assert label not in result, ("ambiguous role", label)
        result[label] = row
    assert len(result) == 14
    return result


def analyze(graph, reports, targets):
    flat = {f"{name}/{path}": value for name, r in reports.items() for path, value in traces(r)}
    scopes = build_scopes(graph, {"saved": flat}, {})
    usable = {k: v for k, v in scopes.items() if v[1] is not None}
    products = {
        scope + "/" + name: row.get("attributes", "")
        for scope, (nodes, _) in scopes.items()
        for name, row in nodes.items()
        if row.get("opcode") in ("dot", "convolution")
    }
    result = {}
    for label, row in targets.items():
        bindings = caller_bindings(scopes, row["context"], row["computation"])
        operands = []
        for i, operand in enumerate(row["operands"]):
            value = trace(usable, row["computation"], operand, bindings, tuple(row["context"]), products)
            for boundary in value["boundaries"]:
                if boundary["reason"] == "polar product anchor":
                    boundary["reason"] = "declared product frontier"
            value.pop("rooted_signature")
            value.pop("signature_scope")
            operands.append(dict(operand_position=i, trace=value))
        result[label] = dict(endpoint=row, operands=operands)
    return dict(
        targets=result, limitations="Declared producer bindings only; all original opaque outputs remain opaque."
    )
