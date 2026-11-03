# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Classify upstream product and collective input frontiers and saved exact-context input coverage."""

import argparse
import json
from pathlib import Path

from compare_upstream_bindings import indexed
from consolidate_compiler_records import identity, traces


def summarize(report, inventory, saved):
    saved = dict(saved, current=report)
    result = {}
    for name, graph in report["artifacts"].items():
        all_traces = [
            (label, path, value)
            for label, source in saved.items()
            for path, value in traces(source["artifacts"][name]["trace"])
        ]
        roots = {}
        for label, path, value in all_traces:
            if value["missing_records"]:
                continue
            row = value["records"][value["root"]]
            roots.setdefault(identity(row), []).append(dict(report=label, path=list(path)))
        polar = {
            r["operation"]: dict(group=group, iteration=r["iteration"], product=r["product"])
            for group, chain in inventory["artifacts"][name]["polar_lineage"]["output_groups"].items()
            for r in chain
        }
        rows = {}
        for label, value in indexed(graph).items():
            frontiers = []
            for boundary in value["boundaries"]:
                row = value["records"][boundary["node"]]
                role = polar.get(row["computation"] + "/" + row["name"])
                coverage = []
                for i, operand in enumerate(row["operands"]):
                    key = identity(dict(row, name=operand, tuple_path=[]))
                    coverage.append(dict(operand_position=i, existing_input_traces=roots.get(key, [])))
                frontiers.append(dict(record=row, boundary=boundary, polar_role=role, declared_input_coverage=coverage))
            rows[label] = frontiers
        result[name] = rows
    return dict(
        graphs=result,
        limitations=[
            "Coverage joins exact graph/caller/computation/instruction/empty tuple path of complete saved trace roots.",
            "Each operand position retained, including repeated operands. Record presence alone is not input coverage.",
            "Covered declared inputs do not establish opaque output semantics or input runtime values.",
            "Source product roles come from dependency-mapped column polar lineage; no text-order matching.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--saved", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        json.loads(args.report.read_text()),
        json.loads(args.inventory.read_text()),
        {p.stem: json.loads(p.read_text()) for p in args.saved},
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
