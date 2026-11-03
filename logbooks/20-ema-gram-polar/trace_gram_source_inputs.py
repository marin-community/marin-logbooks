# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select uncovered upstream operands by saved dependencies and exact input-root coverage."""

from compare_norm_polar_inputs import indexed
from consolidate_compiler_records import declaration, identity, traces
from trace_boundary_inputs import TARGET
from trace_earlier_bindings import analyze

EXPECTED_PRODUCTS = {
    ("down_proj", 1, "X_Xt"),
    ("up_proj", 1, "B_X"),
    ("up_proj", 2, "scaled_A_A"),
    ("up_proj", 3, "X_Xt"),
}


def nonpolar_products(current):
    """Select the three source-scoped products on the two actual normalization input routes."""
    values = indexed(dict(trace=current))
    result = {}
    specifications = {
        "direct_product": (0, "/gram_direction/dot_general"),
        "refresh_product": (1, "/update_gram_state/dot_general"),
        "relative_power": (1, "/gram_direction/relative_gram_power/dot_general"),
    }
    for label, (position, scope) in specifications.items():
        value = values[f"down_proj/normalization/gram_direction/{position}"]
        rows = [value["records"][b["node"]] for b in value["boundaries"]]
        matches = [r for r in rows if r["opcode"] in ("convolution", "dot") and scope + '"' in r["attributes"]]
        assert len(matches) == 1, "Missing or ambiguous source-scoped product on selected operand dependency"
        result[label] = matches[0]
    assert len({identity(r) for r in result.values()}) == 3
    return result


def select_targets(graph, current, reports):
    nonpolar = {identity(r): label for label, r in nonpolar_products(current).items()}
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
            candidates[key][1].add(label)
    result, products = {}, set()
    for row, groups in candidates.values():
        covered = [identity(dict(row, name=n, tuple_path=[])) in roots for n in row["operands"]]
        if all(covered):
            continue
        assert not any(covered), "Partially inspected input set requires explicit position selection"
        assert len({v.split("/")[0] for v in groups}) == 1 and not row["tuple_path"]
        group = next(iter(groups)).split("/")[0]
        if row["opcode"] in ("convolution", "dot"):
            role = polar.get(row["computation"] + "/" + row["name"])
            assert len(row["operands"]) == 2
            if role is None:
                assert group == "down_proj" and identity(row) in nonpolar
                label = "down_proj/source/" + nonpolar[identity(row)]
            else:
                assert role in EXPECTED_PRODUCTS and role[0] == group
                products.add(role)
                label = f"{group}/polar/{role[1]}/{role[2]}"
        else:
            match = TARGET.search(row.get("attributes", ""))
            assert row["opcode"] == "custom-call" and match
            kind = match[1]
            assert kind in ("ConcatBitcast", "AsyncCollectiveDone")
            assert len(row["operands"]) == (4 if kind == "ConcatBitcast" else 10)
            label = f"{min(groups)}/frontier/{kind}"
        assert label not in result, "Ambiguous endpoint within group"
        result[label] = row
    assert products == EXPECTED_PRODUCTS
    assert {"down_proj/source/" + label for label in nonpolar.values()} <= result.keys()
    assert len(result) == 12 and sum(len(r["operands"]) for r in result.values()) == 52
    return result


__all__ = ["analyze", "select_targets"]
