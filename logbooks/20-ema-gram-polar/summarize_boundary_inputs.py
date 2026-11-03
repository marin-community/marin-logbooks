# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize auxiliary endpoint operand traces without inferring call semantics."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from compare_outer_bindings import normalized
from trace_boundary_inputs import TARGET


def identity(row):
    return (tuple(row["context"]), row["computation"], row["name"])


def summarize(report):
    assert report["completed"]
    graphs = {}
    control = {identity(e["endpoint"]): e for e in report["artifacts"]["matrix_untouched"]["trace"]["endpoints"]}
    evaluator_checks = []
    for variant, graph in report["artifacts"].items():
        assert graph["sha256_verified"]
        endpoints = []
        for e in graph["trace"]["endpoints"]:
            operands = []
            for o in e["operand_traces"]:
                t = o["trace"]
                assert not t["missing_records"]
                labels = []
                for b in t["boundaries"]:
                    row = t["records"][b["node"]]
                    m = TARGET.search(row.get("attributes", ""))
                    labels.append(
                        dict(
                            node=b["node"],
                            reason=b["reason"],
                            computation=row["computation"],
                            name=row["name"],
                            context=row["context"],
                            tuple_path=row["tuple_path"],
                            opcode=row.get("opcode"),
                            custom_call_target=m[1] if m else None,
                            polar_role=b.get("role"),
                        )
                    )
                operands.append(
                    dict(
                        position=o["operand_position"],
                        records=len(t["records"]),
                        edges=len(t["edges"]),
                        boundaries=labels,
                    )
                )
                if variant == "matrix_evaluate":
                    expected = control[identity(e["endpoint"])]["operand_traces"][o["operand_position"]]["trace"]
                    equal = normalized(t, variant)[0] == normalized(expected, "matrix_untouched")[0]
                    evaluator_checks.append(equal)
            start_sets = [
                {identity(b) for b in o["boundaries"] if b["custom_call_target"] == "AsyncCollectiveStart"}
                for o in operands
            ]
            common = set.intersection(*start_sets) if start_sets else set()
            if e["custom_call_target"] == "AsyncCollectiveDone":
                assert len(operands) == 10
                assert all(len(s) == 1 for s in start_sets) and len(common) == 1
            compact_operands = [
                dict(
                    position=o["position"],
                    records=o["records"],
                    edges=o["edges"],
                    start_projections=[
                        b["tuple_path"] for b in o["boundaries"] if b["custom_call_target"] == "AsyncCollectiveStart"
                    ],
                    boundary_types=dict(Counter(b["custom_call_target"] or b["opcode"] for b in o["boundaries"])),
                )
                for o in operands
            ]
            endpoints.append(
                dict(
                    endpoint={k: e["endpoint"][k] for k in ("computation", "name", "context", "shape", "operands")},
                    root_uses=e["root_uses"],
                    target=e["custom_call_target"],
                    operands=compact_operands,
                    all_operand_start_identities=[list(x) for x in sorted(set.union(*start_sets))],
                    unique_common_start=(len(common) == 1 and all(s == common for s in start_sets)),
                )
            )
        graphs[variant] = dict(
            endpoints=endpoints,
            endpoint_types=dict(Counter(e["target"] for e in endpoints)),
            operand_positions=sum(len(e["operands"]) for e in endpoints),
            extraction_rounds=len(graph["extraction_rounds"]),
            extracted_bodies=sum(map(len, graph["extraction_rounds"])),
        )
    assert len(evaluator_checks) == 176 and all(evaluator_checks)
    return dict(
        graphs=graphs,
        evaluator_normalized_operand_checks=len(evaluator_checks),
        evaluator_all_match=True,
        limitations=[
            "Common start identity is a declared operand connection, not proof of exact custom-call semantics.",
            "Original endpoint outputs remain opaque. Auxiliary traces also stop at opaque calls and while loops.",
            "Endpoint counts are context-deduplicated; operand and boundary counts include repeated uses.",
            "Capture/substitution endpoints are not paired by text order or list index.",
            "Both numerical full28leaf gates remain failed; no runtime precision, order, values or cause inferred.",
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    raw = a.report.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == a.sha256
    result = summarize(json.loads(raw))
    result["report_sha256"] = a.sha256
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
