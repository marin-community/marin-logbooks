# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect declared collective inputs selected by saved polar operand dependencies."""

from consolidate_compiler_records import traces
from trace_boundary_inputs import TARGET, build_scopes, caller_bindings
from trace_outer_bindings import trace
from trace_start_producers import identity

ROLES = ("down_proj/polar/5/scaled_A_A", "o_proj/polar/2/B_X", "o_proj/polar/3/B_X")


def select_targets(frontier):
    result = {}
    for role in ROLES:
        operands = frontier["targets"][role]["operands"]
        assert sorted(v["operand_position"] for v in operands) == [0, 1]
        value = next(v["trace"] for v in operands if v["operand_position"] == 1)
        assert not value["missing_records"]
        found = []
        for boundary in value["boundaries"]:
            row = value["records"][boundary["node"]]
            match = TARGET.search(row.get("attributes", ""))
            if match:
                assert row["opcode"] == "custom-call" and match[1] in ("AsyncCollectiveDone", "ConcatBitcast")
                found.append(dict(endpoint=row, kind=match[1]))
        assert len(found) == 1, (role, len(found))
        assert len(found[0]["endpoint"]["operands"]) == (10 if found[0]["kind"] == "AsyncCollectiveDone" else 4)
        if role.startswith("down_proj"):
            assert found[0]["kind"] == "AsyncCollectiveDone"
        result[role] = dict(**found[0], original_operand_trace=value)
    return result


def common_start(operands):
    assert sorted(v["operand_position"] for v in operands) == list(range(10))
    found, positions = {}, []
    for operand in sorted(operands, key=lambda v: v["operand_position"]):
        value = operand["trace"]
        assert not value["missing_records"]
        matches = [
            value["records"][b["node"]]
            for b in value["boundaries"]
            if 'custom_call_target="AsyncCollectiveStart"' in value["records"][b["node"]].get("attributes", "")
        ]
        assert len(matches) == 1
        row = matches[0]
        found[identity(row)] = row
        positions.append(dict(operand_position=operand["operand_position"], start_tuple_path=row["tuple_path"]))
    assert len(found) == 1, "Done operands do not reach one common Start identity"
    start = next(iter(found.values()))
    assert len(start["operands"]) == 1
    return start, positions


def analyze(graph, reports, targets):
    flat = {f"{name}/{path}": value for name, r in reports.items() for path, value in traces(r)}
    scopes = build_scopes(graph, {"saved": flat}, {})
    usable = {k: v for k, v in scopes.items() if v[1] is not None}
    products = {
        scope + "/" + name: row.get("attributes", "")
        for scope, (nodes, _) in scopes.items()
        for name, row in nodes.items()
        if row.get("opcode") in ("convolution", "dot")
    }

    def inputs(row):
        bindings = caller_bindings(scopes, row["context"], row["computation"])
        result = []
        for i, operand in enumerate(row["operands"]):
            value = trace(usable, row["computation"], operand, bindings, tuple(row["context"]), products)
            for b in value["boundaries"]:
                if b["reason"] == "polar product anchor":
                    b["reason"] = "declared product frontier"
            value.pop("rooted_signature")
            value.pop("signature_scope")
            result.append(dict(operand_position=i, trace=value))
        return result

    result = {}
    assert set(targets) == set(ROLES)
    for role, target in targets.items():
        operands = inputs(target["endpoint"])
        result[role] = dict(**target, operands=operands)
        if target["kind"] == "AsyncCollectiveDone":
            if any(v["trace"]["missing_records"] for v in operands):
                result[role]["start_match_pending_missing_records"] = True
                continue
            start, positions = common_start(operands)
            result[role]["matched_start"] = dict(record=start, positions=positions, operands=inputs(start))
    return dict(roles=result, limitations="Declared input bindings only; original opaque outputs remain opaque.")
