# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare column-input static records with own controls and replicated-input graphs."""

import argparse
import hashlib
import json
from pathlib import Path

from compare_substitution_compiler import summarize
from trace_first_gram_paths import analyze


def product_fields(graph):
    operations = {r["computation"] + "/" + r["name"]: r for r in graph["selected_operations"]}
    result = {}
    for group, chain in graph["polar_lineage"]["output_groups"].items():
        for item in chain:
            row = operations[item["operation"]]
            parents = [r for r in graph["fusion_callers"] if r["calls"] == row["computation"]]
            assert len(parents) == 1
            result[f"{group}/{item['iteration']}/{item['product']}"] = dict(
                shape=row["shape"],
                operand_shapes=[p["shape"] for p in row["direct_operand_producers"]],
                operand_opcodes=[p["opcode"] for p in row["direct_operand_producers"]],
                operation_attributes=row["attributes"].split(", metadata=", 1)[0],
                operation_backend=row["backend_config"],
                enclosing_fusion_backend=parents[0]["backend_config"],
            )
    assert len(result) == 105
    return result


def compare_pair(left, right):
    a, b = product_fields(left), product_fields(right)
    assert a.keys() == b.keys()
    differences = {field: [k for k in a if a[k][field] != b[k][field]] for field in next(iter(a.values()))}
    return dict(
        matched_positions=105,
        differences=differences,
        affected_groups={field: sorted({k.split("/")[0] for k in rows}) for field, rows in differences.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--prior-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths-output", type=Path, required=True)
    args = parser.parse_args()
    sources = []
    for path, expected in ((args.report, args.report_sha256), (args.prior, args.prior_sha256)):
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        report = json.loads(raw)
        assert report["completed"] and len(report["artifacts"]) == 8
        assert all(g["sha256_verified"] for g in report["artifacts"].values())
        sources.append(report)
    current, prior = sources
    result = summarize(current)
    result["input_report_sha256"] = args.report_sha256
    result["prior_report_sha256"] = args.prior_sha256
    result["cross_job_same_role"] = {
        name: compare_pair(prior["artifacts"][name], graph) for name, graph in current["artifacts"].items()
    }
    supplement = dict(
        artifacts={
            name: dict(
                sha256_verified=graph["sha256_verified"],
                input_sha256=current["manifest"]["artifacts"][name]["sha256"],
                bodies=graph["first_gram_path_bodies"],
            )
            for name, graph in current["artifacts"].items()
        }
    )
    paths = analyze(current, supplement)
    paths["input_report_sha256"] = args.report_sha256
    result["first_gram_path_summary"] = {
        name: {
            group: [
                dict(
                    operand_index=p["operand_index"],
                    collective_opcodes=[p["records"][k]["opcode"] for k in p["collectives"]],
                    missing_records=p["missing_records"],
                    boundaries=p["boundaries"],
                    first_product_fusion_output_path=p["first_product_fusion_output_path"],
                )
                for p in row["operands"]
            ]
            for group, row in groups.items()
        }
        for name, groups in paths["graphs"].items()
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.paths_output.write_text(json.dumps(paths, indent=2) + "\n")


if __name__ == "__main__":
    main()
