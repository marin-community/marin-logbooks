# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect operands of product and ConcatBitcast frontiers reached from start inputs."""

from trace_boundary_inputs import build_scopes, caller_bindings
from trace_outer_bindings import trace


def analyze(graph, outer, boundary, start_report, extra=None):
    combined = dict(outer)
    combined["auxiliary"] = {
        f"{i}/{j}": operand["trace"]
        for i, endpoint in enumerate(boundary["endpoints"])
        for j, operand in enumerate(endpoint["operand_traces"])
    }
    combined["starts"] = {str(i): s["declared_input_trace"] for i, s in enumerate(start_report["starts"])}
    scopes = build_scopes(graph, combined, extra or {})
    usable = {k: v for k, v in scopes.items() if v[1] is not None}
    products = {
        scope + "/" + name: row.get("attributes", "")
        for scope, (nodes, _) in scopes.items()
        for name, row in nodes.items()
        if row.get("opcode") in ("convolution", "dot")
    }
    result = {}
    for selected in start_report["starts"]:
        assert len(selected["uses"]) == 1
        group = selected["uses"][0]["group"]
        assert group not in result
        result[group] = {}
        prior = selected["declared_input_trace"]
        for label in prior["boundaries"]:
            row = prior["records"][label["node"]]
            if label["reason"] == "declared product frontier":
                kind, count = "product", 2
            else:
                assert row["opcode"] == "custom-call" and 'custom_call_target="ConcatBitcast"' in row["attributes"]
                kind, count = "concat_declared_inputs", 4
            assert len(row["operands"]) == count and kind not in result[group]
            bindings = caller_bindings(scopes, row["context"], row["computation"])
            traces = []
            for position, operand in enumerate(row["operands"]):
                value = trace(usable, row["computation"], operand, bindings, tuple(row["context"]), products)
                for b in value["boundaries"]:
                    if b["reason"] == "polar product anchor":
                        b["reason"] = "declared product frontier"
                value.pop("rooted_signature")
                value.pop("signature_scope")
                traces.append(dict(operand_position=position, trace=value))
            result[group][kind] = dict(endpoint=row, operands=traces)
        assert set(result[group]) == (
            {"product", "concat_declared_inputs"} if group in ("down_proj", "o_proj") else {"product"}
        )
    assert set(result) == {"gate_proj", "down_proj", "up_proj", "o_proj"}
    return dict(
        groups=result, limitations="Declared operands only; original ConcatBitcast and Start/Done outputs remain opaque."
    )
