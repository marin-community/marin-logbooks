# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select uncovered Concat inputs through the exact left-power initializer dependency."""

from compare_polar_initializers import indexed
from consolidate_compiler_records import identity, traces
from trace_frontier_operands import analyze

ROLE = "down_proj/left_power/while_initializer/position3/ConcatBitcast"
DEPENDENCY = "down_proj/left_power/while_initializer/0"


def select_targets(graph, current, reports):
    del graph
    value = indexed(dict(trace=current))[DEPENDENCY]
    assert not value["missing_records"]
    root = value["records"][value["root"]]
    assert root["opcode"] == "tuple" and len(root["operands"]) == 12
    edges = [e for e in value["edges"] if e["source"] == value["root"] and e["operand_index"] == 3]
    assert len(edges) == 1
    seen, pending = set(), [edges[0]["target"]]
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        pending.extend(e["target"] for e in value["edges"] if e["source"] == node)
    roots = {
        identity(v["records"][v["root"]]) for r in reports.values() for _, v in traces(r) if not v["missing_records"]
    }
    candidates = []
    for boundary in value["boundaries"]:
        row = value["records"][boundary["node"]]
        if boundary["node"] not in seen or 'custom_call_target="ConcatBitcast"' not in row.get("attributes", ""):
            continue
        coverage = [identity(dict(row, name=n, tuple_path=[])) in roots for n in row["operands"]]
        if all(coverage):
            continue
        assert not any(coverage), "Partial input coverage requires an explicit narrower selector"
        assert row["opcode"] == "custom-call" and len(row["operands"]) == 4 and not row["tuple_path"]
        candidates.append(row)
    assert len(candidates) == 1, "Missing or ambiguous uncovered Concat on initializer position3"
    return {ROLE: candidates[0]}


__all__ = ["DEPENDENCY", "ROLE", "analyze", "select_targets"]
