# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Classify fifth-step operand frontiers and saved exact-context input coverage."""

import argparse
import json
from pathlib import Path

from compare_step5_gram_inputs import indexed
from consolidate_compiler_records import identity, traces


def summarize(report, inventory, saved):
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
                if row["opcode"] in ("dot", "convolution"):
                    assert label == "down_proj/polar/5/X_Xt/0"
                    assert role in [dict(group="down_proj", iteration=i, product="B_X") for i in (3, 4)]
                coverage = []
                for i, operand in enumerate(row["operands"]):
                    key = identity(dict(row, name=operand, tuple_path=[]))
                    coverage.append(dict(operand_position=i, existing_input_traces=roots.get(key, [])))
                frontiers.append(dict(record=row, boundary=boundary, polar_role=role, declared_input_coverage=coverage))
            rows[label] = frontiers
        assert sorted(r["polar_role"]["iteration"] for r in rows["down_proj/polar/5/X_Xt/0"] if r["polar_role"]) == [
            3,
            4,
        ]
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
