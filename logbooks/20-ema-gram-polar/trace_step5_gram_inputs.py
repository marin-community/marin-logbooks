# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select fifth-step Gram products by mapped polar role and unique entry caller."""

from consolidate_compiler_records import declaration, traces
from trace_boundary_inputs import build_scopes, caller_bindings
from trace_frontier_operands import analyze

GROUPS = ("down_proj", "up_proj", "o_proj")


def select_targets(graph, reports):
    flat = {f"{label}/{path}": t for label, report in reports.items() for path, t in traces(report)}
    scopes = build_scopes(graph, {"saved": flat}, {})
    result = {}
    for group in GROUPS:
        roles = [
            r for r in graph["polar_lineage"]["output_groups"][group] if r["iteration"] == 5 and r["product"] == "X_Xt"
        ]
        assert len(roles) == 1
        scope, name = roles[0]["operation"].split("/", 1)
        candidates = [r for r in graph["selected_operations"] if r["computation"] == scope and r["name"] == name]
        assert len(candidates) == 1
        selected = candidates[0]
        assert selected["opcode"] in ("dot", "convolution") and len(selected["operands"]) == 2
        assert "/gram_direction/" in selected["op_name"]
        callers = [r for r in graph["fusion_callers"] if r["calls"] == scope]
        assert len(callers) == 1
        caller = callers[0]
        # Require an actual entry instruction, not an inferred caller from naming.
        entries = graph["lineage_diagnostics"]["entry_instructions"]
        assert sum("%" + caller["name"] + " =" in line for line in entries) == 1
        context = [caller["computation"] + "/" + caller["name"]]
        caller_bindings(scopes, context, scope)
        row = {k: selected[k] for k in ("name", "shape", "opcode", "operands", "attributes", "computation")}
        row.update(context=context, tuple_path=[], parameter_index=None)
        found = [
            r for value in flat.values() for r in value["records"] if r["computation"] == scope and r["name"] == name
        ]
        for saved in found:
            assert saved["context"] == context and not saved["tuple_path"]
            assert declaration(saved) == declaration(row)
        if group == "down_proj":
            # Its current dependency-selected Concat inputs must reach this product.
            down = reports["down-concat"]["targets"]
            assert len(down) == 1
            for operand in next(iter(down.values()))["operands"]:
                value = operand["trace"]
                products = [
                    value["records"][b["node"]]
                    for b in value["boundaries"]
                    if value["records"][b["node"]]["opcode"] in ("dot", "convolution")
                ]
                assert len(products) == 1
                assert products[0]["computation"] == scope and products[0]["name"] == name
                assert products[0]["context"] == context
        result[f"{group}/polar/5/X_Xt"] = row
    return result


__all__ = ["GROUPS", "analyze", "select_targets"]
