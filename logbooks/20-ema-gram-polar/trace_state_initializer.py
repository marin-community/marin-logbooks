# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select uncovered upstream operands by saved dependencies and exact input-root coverage."""

from compare_gram_source_inputs import indexed
from consolidate_compiler_records import declaration, identity, traces
from trace_boundary_inputs import TARGET, build_scopes, caller_bindings
from trace_earlier_bindings import analyze as analyze_bindings
from trace_initializer_tuple import trace as trace_initializer

EXPECTED_PRODUCTS = {("up_proj", 1, "scaled_A_A"), ("up_proj", 2, "X_Xt")}


def nonpolar_products(current):
    """Select source products on exact saved operand dependencies, without assigning polar roles."""
    values = indexed(dict(trace=current))
    specifications = {
        "down_proj/source/refresh_product": ("down_proj/source/direct_product/0", "/update_gram_state/dot_general"),
        "down_proj/source/relative_power": (
            "down_proj/source/direct_product/0",
            "/gram_direction/relative_gram_power/dot_general",
        ),
        "up_proj/source/direct_product": (
            "up_proj/opaque/ConcatBitcast/0/frontier/ConcatBitcast/0/frontier/ConcatBitcast/0",
            "/gram_direction/dot_general",
        ),
    }
    result = {}
    for label, (dependency, scope) in specifications.items():
        value = values[dependency]
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
        assert len({v.split("/")[0] for v in groups}) == 1
        assert not row["tuple_path"] or (row["opcode"] == "while" and row["tuple_path"] == [1])
        group = next(iter(groups)).split("/")[0]
        if row["opcode"] in ("convolution", "dot"):
            role = polar.get(row["computation"] + "/" + row["name"])
            assert len(row["operands"]) == 2
            if role is None:
                assert identity(row) in nonpolar
                label = nonpolar[identity(row)]
                assert label.split("/")[0] == group
            else:
                assert role in EXPECTED_PRODUCTS and role[0] == group
                products.add(role)
                label = f"{group}/polar/{role[1]}/{role[2]}"
        elif row["opcode"] == "while":
            assert group == "down_proj" and groups == {"down_proj/source/relative_power/0"}
            assert row["tuple_path"] == [1] and len(row["operands"]) == 1
            assert "/relative_gram_power/shard_map/while" in row["attributes"]
            label = "down_proj/source/relative_power/while_initializer"
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
    assert set(nonpolar.values()) <= result.keys()
    assert len(result) == 9 and sum(len(r["operands"]) for r in result.values()) == 35
    return result


def analyze(graph, reports, targets):
    key = "down_proj/source/relative_power/while_initializer"
    result = analyze_bindings(graph, reports, {k: v for k, v in targets.items() if k != key})
    row = targets[key]
    assert row["opcode"] == "while" and row["tuple_path"] == [1] and len(row["operands"]) == 1
    flat = {f"{label}/{path}": value for label, report in reports.items() for path, value in traces(report)}
    scopes = build_scopes(graph, {"saved": flat}, {})
    usable = {k: v for k, v in scopes.items() if v[1] is not None}
    products = {
        scope + "/" + name: node.get("attributes", "")
        for scope, (nodes, _) in scopes.items()
        for name, node in nodes.items()
        if node.get("opcode") in ("dot", "convolution")
    }
    bindings = caller_bindings(scopes, row["context"], row["computation"])
    value = trace_initializer(usable, row["computation"], row["operands"][0], bindings, tuple(row["context"]), products)
    for boundary in value["boundaries"]:
        if boundary["reason"] == "polar product anchor":
            boundary["reason"] = "declared product frontier"
    value.pop("rooted_signature")
    value.pop("signature_scope")
    result["targets"][key] = dict(endpoint=row, operands=[dict(operand_position=0, trace=value)])
    return result


__all__ = ["analyze", "select_targets"]
