# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select uncovered upstream operands by saved dependencies and exact input-root coverage."""

from compare_upstream_bindings import indexed
from consolidate_compiler_records import declaration, identity, traces
from trace_boundary_inputs import TARGET
from trace_earlier_bindings import analyze

EXPECTED_PRODUCTS = {
    ("down_proj", 1, "scaled_A_A"),
    ("up_proj", 3, "scaled_A_A"),
    ("up_proj", 2, "B_X"),
}


def normalization_product(current):
    """Require the actual selected normalization input to reach one repeated product output."""
    found = []
    values = indexed(dict(trace=current))
    for position in range(4):
        value = values[f"down_proj/opaque/ConcatBitcast/{position}"]
        rows, edges = value["records"], value["edges"]
        ars = [(i, r) for i, r in enumerate(rows) if r["opcode"] == "all-reduce"]
        assert len(ars) == 1
        index, ar = ars[0]
        assert len(ar["tuple_path"]) == 1
        outgoing = [e for e in edges if e["source"] == index]
        assert len(outgoing) == 1 and outgoing[0]["operand_index"] == ar["tuple_path"][0]
        incoming = [rows[e["source"]] for e in edges if e["target"] == index]
        assert len(incoming) == 1 and incoming[0]["opcode"] == "get-tuple-element"
        assert "/vmap(jit(norm))/reduce_sum" in incoming[0]["attributes"]
        seen, todo = set(), [outgoing[0]["target"]]
        while todo:
            node = todo.pop()
            if node in seen:
                continue
            seen.add(node)
            todo.extend(e["target"] for e in edges if e["source"] == node)
        products = [i for i in seen if rows[i]["opcode"] in ("convolution", "dot")]
        assert len(products) == 1
        product = products[0]
        assert "/gram_direction/dot_general" in rows[product]["attributes"]
        squares = []
        for i in seen:
            es = [e for e in edges if e["source"] == i]
            if rows[i]["opcode"] == "multiply" and len(es) == 2:
                if sorted(e["operand_index"] for e in es) == [0, 1] and all(e["target"] == product for e in es):
                    squares.append(i)
        assert len(squares) == 1
        reductions = [
            e["source"] for e in edges if e["target"] == squares[0] and rows[e["source"]]["opcode"] == "reduce"
        ]
        assert len(reductions) == 1 and "dimensions={1,2}" in rows[reductions[0]]["attributes"]
        found.append(rows[product])
    assert len({identity(r) for r in found}) == 1
    return found[0]


def select_targets(graph, current, reports):
    norm_product = normalization_product(current)
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
                assert group == "down_proj" and identity(row) == identity(norm_product)
                label = "down_proj/normalization/gram_direction"
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
    assert "down_proj/normalization/gram_direction" in result
    assert len(result) == 11 and sum(len(r["operands"]) for r in result.values()) == 48
    return result


__all__ = ["analyze", "select_targets"]
