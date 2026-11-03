# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select opaque Concats through fifth-step Gram operand-zero dependencies."""

from trace_frontier_operands import analyze

GROUPS = ("down_proj", "up_proj", "o_proj")


def select_targets(step5):
    result = {}
    for group in GROUPS:
        target = step5["targets"][f"{group}/polar/5/X_Xt"]
        operands = [r for r in target["operands"] if r["operand_position"] == 0]
        assert len(operands) == 1
        value = operands[0]["trace"]
        assert not value["missing_records"]
        root = value["records"][value["root"]]
        assert root["name"] == target["endpoint"]["operands"][0]
        assert root["context"] == target["endpoint"]["context"]
        assert root["computation"] == target["endpoint"]["computation"]
        candidates = [
            value["records"][b["node"]]
            for b in value["boundaries"]
            if 'custom_call_target="ConcatBitcast"' in value["records"][b["node"]].get("attributes", "")
        ]
        assert len(candidates) == 1
        row = candidates[0]
        assert row["opcode"] == "custom-call" and len(row["operands"]) == 4 and not row["tuple_path"]
        result[f"{group}/polar/5/X_Xt/operand0/ConcatBitcast"] = row
    return result


__all__ = ["GROUPS", "analyze", "select_targets"]
