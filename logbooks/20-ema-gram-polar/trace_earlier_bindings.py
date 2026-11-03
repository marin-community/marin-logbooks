# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect uncovered product and collective inputs behind earlier B@X operands."""

from compare_earlier_bx_inputs import indexed
from consolidate_compiler_records import identity, traces
from trace_boundary_inputs import TARGET
from trace_frontier_collectives import common_start
from trace_frontier_operands import analyze as trace_inputs


def select_targets(graph, current):
    result = {}
    for label, value in indexed(dict(trace=current)).items():
        group, _, iteration, _, position = label.split("/")
        kinds = []
        for boundary in value["boundaries"]:
            row = value["records"][boundary["node"]]
            assert not row["tuple_path"]
            if row["opcode"] in ("dot", "convolution"):
                matches = [
                    r
                    for r in graph["polar_lineage"]["output_groups"][group]
                    if r["operation"] == row["computation"] + "/" + row["name"]
                ]
                assert len(matches) == 1 and matches[0]["iteration"] == int(iteration)
                assert matches[0]["product"] == "scaled_A_A" and len(row["operands"]) == 2
                kind = "scaled_A_A"
            else:
                match = TARGET.search(row.get("attributes", ""))
                assert row["opcode"] == "custom-call" and match
                kind = match[1]
                assert kind in ("ConcatBitcast", "AsyncCollectiveDone")
                assert len(row["operands"]) == (4 if kind == "ConcatBitcast" else 10)
            key = f"{group}/polar/{iteration}/{kind}"
            assert key not in result
            result[key] = row
            kinds.append(kind)
        assert sorted(kinds) == (["ConcatBitcast", "scaled_A_A"] if position == "0" else ["AsyncCollectiveDone"])
    assert len(result) == 6 and sum(len(r["operands"]) for r in result.values()) == 32
    return result


def assert_uncovered(reports, targets):
    roots = {
        identity(v["records"][v["root"]]) for r in reports.values() for _, v in traces(r) if not v["missing_records"]
    }
    for row in targets.values():
        for operand in row["operands"]:
            assert identity(dict(row, name=operand, tuple_path=[])) not in roots


def analyze(graph, reports, targets):
    result = trace_inputs(graph, reports, targets)
    for label, target in result["targets"].items():
        if not label.endswith("/AsyncCollectiveDone"):
            continue
        if any(v["trace"]["missing_records"] for v in target["operands"]):
            target["start_match_pending_missing_records"] = True
            continue
        start, positions = common_start(target["operands"])
        # Preserve all ten projections; this is an auxiliary input trace, not custom-call semantics.
        extra = trace_inputs(graph, reports, {label: start})["targets"][label]
        target["matched_start"] = dict(record=start, positions=positions, operands=extra["operands"])
    return result
