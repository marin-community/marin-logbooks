# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect declared inputs behind selected polar operand boundaries."""

from trace_boundary_inputs import build_scopes, caller_bindings
from trace_outer_bindings import trace
from trace_start_producers import identity


def analyze(graph, outer, boundary, starts, product_inputs, extra=None):
    combined = dict(outer)
    combined["auxiliary"] = {
        f"{i}/{j}": operand["trace"]
        for i, endpoint in enumerate(boundary["endpoints"])
        for j, operand in enumerate(endpoint["operand_traces"])
    }
    combined["starts"] = {str(i): s["declared_input_trace"] for i, s in enumerate(starts["starts"])}
    combined["products"] = {
        f'{group}/{kind}/{v["operand_position"]}': v["trace"]
        for group, kinds in product_inputs["groups"].items()
        for kind, endpoint in kinds.items()
        for v in endpoint["operands"]
    }
    scopes = build_scopes(graph, combined, extra or {})
    usable = {k: v for k, v in scopes.items() if v[1] is not None}
    products = {
        scope + "/" + name: row.get("attributes", "")
        for scope, (nodes, _) in scopes.items()
        for name, row in nodes.items()
        if row.get("opcode") in ("convolution", "dot")
    }

    def declared_inputs(row):
        bindings = caller_bindings(scopes, row["context"], row["computation"])
        result = []
        for position, operand in enumerate(row["operands"]):
            value = trace(usable, row["computation"], operand, bindings, tuple(row["context"]), products)
            for b in value["boundaries"]:
                if b["reason"] == "polar product anchor":
                    b["reason"] = "declared product frontier"
            value.pop("rooted_signature")
            value.pop("signature_scope")
            result.append(dict(operand_position=position, trace=value))
        return result

    result = {}
    for group in ("gate_proj", "down_proj", "up_proj", "o_proj"):
        prior = [
            v["trace"] for v in product_inputs["groups"][group]["product"]["operands"] if v["operand_position"] == 1
        ]
        assert len(prior) == 1 and not prior[0]["missing_records"]
        kind, count = ("AsyncCollectiveDone", 10) if group in ("down_proj", "o_proj") else ("ConcatBitcast", 4)
        candidates = [
            prior[0]["records"][b["node"]]
            for b in prior[0]["boundaries"]
            if f'custom_call_target="{kind}"' in prior[0]["records"][b["node"]].get("attributes", "")
        ]
        assert len(candidates) == 1
        endpoint = candidates[0]
        assert endpoint["opcode"] == "custom-call" and len(endpoint["operands"]) == count
        operands = declared_inputs(endpoint)
        result[group] = dict(endpoint=endpoint, kind=kind, operands=operands)
        if kind == "AsyncCollectiveDone":
            if any(v["trace"]["missing_records"] for v in operands):
                result[group]["start_match_pending_missing_records"] = True
                continue
            matches, positions = {}, []
            for operand in operands:
                value = operand["trace"]
                found = [
                    value["records"][b["node"]]
                    for b in value["boundaries"]
                    if 'custom_call_target="AsyncCollectiveStart"' in value["records"][b["node"]].get("attributes", "")
                ]
                assert len(found) == 1
                row = found[0]
                matches[identity(row)] = row
                positions.append(dict(operand_position=operand["operand_position"], start_tuple_path=row["tuple_path"]))
            assert len(matches) == 1
            start = next(iter(matches.values()))
            assert len(start["operands"]) == 1
            result[group]["matched_start"] = dict(record=start, positions=positions, operands=declared_inputs(start))
    return dict(
        groups=result,
        limitations="Auxiliary declared inputs only. All original Start/Done/Concat outputs remain opaque.",
    )
