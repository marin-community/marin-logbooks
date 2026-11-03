# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select uncovered upstream operands by saved dependencies and exact input-root coverage."""

from compare_earlier_bindings import indexed
from consolidate_compiler_records import declaration, identity, traces
from trace_boundary_inputs import TARGET
from trace_earlier_bindings import analyze

EXPECTED_PRODUCTS = {
    ("down_proj", 2, "X_Xt"),
    ("down_proj", 1, "B_X"),
    ("up_proj", 4, "X_Xt"),
    ("up_proj", 3, "B_X"),
}


def select_targets(graph, current, reports):
    roots = {
        identity(v["records"][v["root"]]) for r in reports.values() for _, v in traces(r) if not v["missing_records"]
    }
    polar = {
        r["operation"]: (group, r["iteration"], r["product"])
        for group, chain in graph["polar_lineage"]["output_groups"].items()
        for r in chain
    }
    candidates = {}
    for label, value in indexed(dict(trace=current)).items():
        for b in value["boundaries"]:
            row = value["records"][b["node"]]
            key = identity(row)
            if key in candidates:
                assert declaration(candidates[key][0]) == declaration(row)
            else:
                candidates[key] = (row, set())
            candidates[key][1].add(label.split("/")[0])
    result, products = {}, set()
    for row, groups in candidates.values():
        covered = [identity(dict(row, name=n, tuple_path=[])) in roots for n in row["operands"]]
        if all(covered):
            continue
        assert not any(covered), "Partially inspected input set requires explicit position selection"
        assert len(groups) == 1 and not row["tuple_path"]
        group = next(iter(groups))
        if row["opcode"] in ("convolution", "dot"):
            role = polar[row["computation"] + "/" + row["name"]]
            assert role in EXPECTED_PRODUCTS and role[0] == group and len(row["operands"]) == 2
            products.add(role)
            label = f"{group}/polar/{role[1]}/{role[2]}"
        else:
            match = TARGET.search(row.get("attributes", ""))
            assert row["opcode"] == "custom-call" and match
            kind = match[1]
            assert kind in ("ConcatBitcast", "AsyncCollectiveDone")
            assert len(row["operands"]) == (4 if kind == "ConcatBitcast" else 10)
            label = f"{group}/opaque/{kind}"
        assert label not in result, "Ambiguous endpoint within group"
        result[label] = row
    assert products == EXPECTED_PRODUCTS
    assert len(result) == 8 and sum(len(r["operands"]) for r in result.values()) == 36
    return result


__all__ = ["analyze", "select_targets"]
