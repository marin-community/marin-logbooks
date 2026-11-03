# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select the opaque down ConcatBitcast by its saved Start-input dependency."""

from trace_frontier_operands import analyze

ROLE = "down_proj/polar/5/scaled_A_A/start_input/ConcatBitcast"


def select_targets(collectives):
    prior = collectives["roles"]["down_proj/polar/5/scaled_A_A"]
    assert prior["kind"] == "AsyncCollectiveDone"
    operands = prior["matched_start"]["operands"]
    assert len(operands) == 1 and operands[0]["operand_position"] == 0
    value = operands[0]["trace"]
    assert not value["missing_records"]
    candidates = [
        value["records"][b["node"]]
        for b in value["boundaries"]
        if 'custom_call_target="ConcatBitcast"' in value["records"][b["node"]].get("attributes", "")
    ]
    assert len(candidates) == 1
    row = candidates[0]
    assert row["opcode"] == "custom-call" and len(row["operands"]) == 4
    return {ROLE: row}


__all__ = ["ROLE", "analyze", "select_targets"]
