# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare dependency-selected collective inputs without aligning different endpoint kinds."""

import argparse
import hashlib
import json
from pathlib import Path

from compare_outer_bindings import normalized, rooted_differences
from compare_start_producers import canonical_trace
from trace_frontier_collectives import ROLES, common_start


def indexed(result):
    roles = result["trace"]["roles"]
    assert set(roles) == set(ROLES)
    output = {}
    for role, value in roles.items():
        assert not value.get("start_match_pending_missing_records")
        count = 10 if value["kind"] == "AsyncCollectiveDone" else 4
        assert value["kind"] in ("AsyncCollectiveDone", "ConcatBitcast")
        assert sorted(v["operand_position"] for v in value["operands"]) == list(range(count))
        operands = {f"declared_input/{v['operand_position']}": v["trace"] for v in value["operands"]}
        if value["kind"] == "AsyncCollectiveDone":
            start, positions = common_start(value["operands"])
            assert start == value["matched_start"]["record"] and positions == value["matched_start"]["positions"]
            assert len(value["matched_start"]["operands"]) == 1
            assert value["matched_start"]["operands"][0]["operand_position"] == 0
            operands["start_input/0"] = value["matched_start"]["operands"][0]["trace"]
        else:
            assert "matched_start" not in value
        for key, trace in operands.items():
            assert not trace["missing_records"]
            endpoint = value["endpoint"] if key.startswith("declared") else value["matched_start"]["record"]
            position = int(key.rsplit("/", 1)[1])
            row = trace["records"][trace["root"]]
            assert row["context"] == endpoint["context"] and row["computation"] == endpoint["computation"]
            assert row["name"] == endpoint["operands"][position]
        output[role] = (value["kind"], operands)
    return output


def compare(report):
    assert report["completed"]
    reference = indexed(report["artifacts"]["matrix_untouched"])
    result = {}
    for name, artifact in report["artifacts"].items():
        assert artifact["sha256_verified"]
        result[name] = {}
        for role, (kind, operands) in indexed(artifact).items():
            base_kind, base_operands = reference[role]
            aligned = kind == base_kind
            comparisons = {}
            for key, t in operands.items():
                value = canonical_trace(t)
                sig = normalized(value, name)[0]
                out = dict(normalized_signature=sig, records=len(t["records"]), edges=len(t["edges"]))
                if aligned:
                    base = canonical_trace(base_operands[key])
                    out.update(
                        same_normalized_signature_as_untouched=sig == normalized(base, "matrix_untouched")[0],
                        rooted_differences=rooted_differences(base, value, name),
                    )
                else:
                    out["comparison"] = "Not aligned: selected endpoint kind differs. No operand pairing assumed."
                out["boundaries"] = [
                    dict(
                        reason=b["reason"],
                        record={
                            k: value["records"][b["node"]][k]
                            for k in ("context", "computation", "name", "tuple_path", "shape", "opcode")
                        },
                    )
                    for b in value["boundaries"]
                ]
                comparisons[key] = out
            result[name][role] = dict(
                kind=kind, untouched_kind=base_kind, endpoint_kind_aligned=aligned, operands=comparisons
            )
    assert all(
        role["endpoint_kind_aligned"]
        and all(v["same_normalized_signature_as_untouched"] for v in role["operands"].values())
        for role in result["matrix_evaluate"].values()
    )
    return dict(
        graphs=result,
        limitations=[
            "Endpoint roles derive from saved output dependencies, caller contexts and polar chain roles.",
            "Different endpoint kinds are retained without pairing operand positions across kinds.",
            "Only declared input bindings; all original custom-call outputs remain opaque.",
            "Normalized shapes, attributes, constants, channel/chain IDs and ordered edges retained.",
            "Difference walker stops at unaligned subtrees; recursive hash does not encode sharing.",
            "Both full28leaf numerical substitution gates remain failed. No values, precision, order or cause claim.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.sha256
    args.output.write_text(json.dumps(compare(json.loads(raw)), indent=2) + "\n")


if __name__ == "__main__":
    main()
