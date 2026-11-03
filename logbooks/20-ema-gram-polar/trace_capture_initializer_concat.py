# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select the shared inner Concat through four saved all-gather dependencies."""

from compare_left_initializer_concat import indexed
from consolidate_compiler_records import identity, traces
from trace_frontier_operands import analyze

ROLE = "down_proj/left_power/while_initializer/position3/inner_ConcatBitcast"


def select_targets(graph, current, reports):
    del graph
    roots = {
        identity(v["records"][v["root"]]) for r in reports.values() for _, v in traces(r) if not v["missing_records"]
    }
    candidates = []
    for value in indexed(dict(trace=current)).values():
        assert len(value["boundaries"]) == 1
        node = value["boundaries"][0]["node"]
        row = value["records"][node]
        assert row["opcode"] == "custom-call" and 'custom_call_target="ConcatBitcast"' in row["attributes"]
        assert len(row["operands"]) == 4 and not row["tuple_path"]
        incoming = [e for e in value["edges"] if e["target"] == node]
        assert len(incoming) == 1 and incoming[0]["operand_index"] == 0
        assert value["records"][incoming[0]["source"]]["opcode"] == "all-gather"
        assert all(identity(dict(row, name=n, tuple_path=[])) not in roots for n in row["operands"])
        candidates.append(row)
    assert len(candidates) == 4 and len({identity(row) for row in candidates}) == 1
    return {ROLE: candidates[0]}


__all__ = ["ROLE", "analyze", "select_targets"]
