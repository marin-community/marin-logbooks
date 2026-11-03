# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select uncovered upstream operands by saved dependencies and exact input-root coverage."""

from compare_state_initializer import indexed
from consolidate_compiler_records import identity, traces
from trace_boundary_inputs import TARGET, build_scopes, caller_bindings
from trace_earlier_bindings import analyze as analyze_bindings
from trace_initializer_tuple import trace as trace_initializer

WHILE_DEPENDENCIES = {
    "down_proj/left_power/while_initializer": ("down_proj/source/relative_power/0", [1]),
    "down_proj/right_power/earlier_while_initializer": ("down_proj/source/relative_power/while_initializer/0", [6]),
}


def select_targets(graph, current, reports):
    values = indexed(dict(trace=current))
    specifications = {
        "up_proj/polar/1/X_Xt": ("up_proj/polar/1/B_X/0/frontier/ConcatBitcast/0", "product"),
        "up_proj/polar/1/scaled_A_A/AsyncCollectiveDone": ("up_proj/polar/1/scaled_A_A/1", "AsyncCollectiveDone"),
        **{label: (dependency, "while") for label, (dependency, _) in WHILE_DEPENDENCIES.items()},
    }
    roots = {
        identity(v["records"][v["root"]]) for r in reports.values() for _, v in traces(r) if not v["missing_records"]
    }
    result = {}
    for label, (dependency, kind) in specifications.items():
        value = values[dependency]
        candidates = []
        for boundary in value["boundaries"]:
            row = value["records"][boundary["node"]]
            match = TARGET.search(row.get("attributes", ""))
            if kind == "product" and row["opcode"] in ("dot", "convolution"):
                candidates.append(row)
            elif kind == "while" and row["opcode"] == "while":
                candidates.append(row)
            elif kind == "AsyncCollectiveDone" and match and match[1] == kind:
                candidates.append(row)
        assert len(candidates) == 1, "Missing or ambiguous endpoint on exact saved output dependency"
        row = candidates[0]
        if kind == "product":
            roles = [
                (group, r["iteration"], r["product"])
                for group, chain in graph["polar_lineage"]["output_groups"].items()
                for r in chain
                if r["operation"] == row["computation"] + "/" + row["name"]
            ]
            assert roles == [("up_proj", 1, "X_Xt")] and len(row["operands"]) == 2 and not row["tuple_path"]
        elif kind == "while":
            assert row["tuple_path"] == WHILE_DEPENDENCIES[label][1] and len(row["operands"]) == 1
            assert "/relative_gram_power/shard_map/while" in row["attributes"]
        else:
            assert row["opcode"] == "custom-call" and len(row["operands"]) == 10 and not row["tuple_path"]
        assert all(identity(dict(row, name=n, tuple_path=[])) not in roots for n in row["operands"])
        result[label] = row
    assert len({identity(row) for row in result.values()}) == 4
    assert sum(len(row["operands"]) for row in result.values()) == 14
    return result


def analyze(graph, reports, targets):
    result = analyze_bindings(graph, reports, {k: v for k, v in targets.items() if k not in WHILE_DEPENDENCIES})
    flat = {f"{label}/{path}": value for label, report in reports.items() for path, value in traces(report)}
    scopes = build_scopes(graph, {"saved": flat}, {})
    usable = {k: v for k, v in scopes.items() if v[1] is not None}
    products = {
        scope + "/" + name: node.get("attributes", "")
        for scope, (nodes, _) in scopes.items()
        for name, node in nodes.items()
        if node.get("opcode") in ("dot", "convolution")
    }
    for key, (_, projection) in WHILE_DEPENDENCIES.items():
        row = targets[key]
        assert row["opcode"] == "while" and row["tuple_path"] == projection and len(row["operands"]) == 1
        bindings = caller_bindings(scopes, row["context"], row["computation"])
        value = trace_initializer(
            usable, row["computation"], row["operands"][0], bindings, tuple(row["context"]), products
        )
        for boundary in value["boundaries"]:
            if boundary["reason"] == "polar product anchor":
                boundary["reason"] = "declared product frontier"
        value.pop("rooted_signature")
        value.pop("signature_scope")
        result["targets"][key] = dict(endpoint=row, operands=[dict(operand_position=0, trace=value)])
    return result


__all__ = ["analyze", "select_targets"]
