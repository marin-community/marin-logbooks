# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select earlier B@X frontiers from four saved Concat input dependencies."""

from consolidate_compiler_records import declaration, identity
from trace_frontier_operands import analyze

ROLES = (("down_proj", 2), ("up_proj", 4))


def select_targets(graph, current):
    result = {}
    for group, iteration in ROLES:
        role = [
            r
            for r in graph["polar_lineage"]["output_groups"][group]
            if r["iteration"] == iteration and r["product"] == "B_X"
        ]
        assert len(role) == 1
        target = current["targets"][f"{group}/polar/5/X_Xt/operand0/ConcatBitcast"]
        assert sorted(r["operand_position"] for r in target["operands"]) == [0, 1, 2, 3]
        found = []
        for operand in target["operands"]:
            value = operand["trace"]
            assert not value["missing_records"]
            candidates = [
                value["records"][b["node"]]
                for b in value["boundaries"]
                if value["records"][b["node"]]["opcode"] in ("dot", "convolution")
            ]
            assert len(candidates) == 1
            row = candidates[0]
            assert row["computation"] + "/" + row["name"] == role[0]["operation"]
            assert not row["tuple_path"] and len(row["operands"]) == 2
            found.append(row)
        assert len({identity(r) for r in found}) == 1
        assert all(declaration(r) == declaration(found[0]) for r in found)
        result[f"{group}/polar/{iteration}/B_X"] = found[0]
    return result


__all__ = ["ROLES", "analyze", "select_targets"]
