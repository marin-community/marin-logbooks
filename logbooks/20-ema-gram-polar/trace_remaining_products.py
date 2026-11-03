# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select uncovered products from exact reconciled output dependencies."""

import json

from consolidate_compiler_records import declaration, identity, traces
from trace_frontier_operands import analyze

EXPECTED_ROLES = {
    ("o_proj", 1, "B_X"),
    ("o_proj", 2, "scaled_A_A"),
    *(("o_proj", step, kind) for step in (3, 4) for kind in ("X_Xt", "scaled_A_A")),
    *(("down_proj", step, "scaled_A_A") for step in (3, 4)),
    *(("gate_proj", step, kind) for step in (4, 5) for kind in ("X_Xt", "scaled_A_A")),
    ("up_proj", 5, "scaled_A_A"),
}


def select_targets(graph, reports, coverage):
    saved = {(label, tuple(path)): value for label, report in reports.items() for path, value in traces(report)}
    roots = {identity(value["records"][value["root"]]) for value in saved.values() if not value["missing_records"]}
    polar = {
        row["operation"]: (group, row["iteration"], row["product"])
        for group, chain in graph["polar_lineage"]["output_groups"].items()
        for row in chain
    }
    targets, roles = {}, set()
    for gap in coverage["gaps"]:
        if gap["category"] != "product":
            continue
        assert gap["projections"] == [[]] and gap["uncovered_positions"] == [0, 1]
        candidates = []
        dependencies = [json.loads(origin) for origin in gap["output_dependencies"]]
        assert dependencies
        for origin in dependencies:
            value = saved[origin["report"], tuple(origin["path"])]
            rows = [
                value["records"][b["node"]]
                for b in value["boundaries"]
                if json.loads(identity(value["records"][b["node"]], False)) == gap["identity"]
            ]
            assert len(rows) == 1, "Missing or duplicate product on exact saved dependency"
            candidates.extend(rows)
        row = candidates[0]
        assert all(identity(r) == identity(row) and declaration(r) == declaration(row) for r in candidates)
        assert declaration(row) == gap["declaration"] and not row["tuple_path"]
        assert row["opcode"] in ("dot", "convolution") and len(row["operands"]) == 2
        assert all(identity(dict(row, name=n, tuple_path=[])) not in roots for n in row["operands"])
        role = polar.get(row["computation"] + "/" + row["name"])
        if role is not None:
            assert role in EXPECTED_ROLES and role not in roles
            assert gap["polar_role"] == dict(group=role[0], iteration=role[1], product=role[2])
            roles.add(role)
            label = f"{role[0]}/polar/{role[1]}/{role[2]}"
        else:
            assert gap["polar_role"] is None and "/gram_direction/dot_general" in row["attributes"]
            assert any("o_proj/polar/2/B_X" in origin["path"] for origin in dependencies)
            assert all(any(str(part).startswith("o_proj/") for part in origin["path"]) for origin in dependencies)
            label = "o_proj/dependency/nonpolar_gram_direction"
        assert label not in targets, "Duplicate product role"
        targets[label] = row
    assert roles == EXPECTED_ROLES and len(targets) == 14
    return dict(sorted(targets.items()))


__all__ = ["analyze", "select_targets"]
