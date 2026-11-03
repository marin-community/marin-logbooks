# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize normalization coverage without interpreting opaque collective results."""

import argparse
import hashlib
import json
from pathlib import Path

from compiler_normalization_closure import missing_scopes


def summarize(source):
    graphs, fields = {}, {}
    for name, artifact in source["artifacts"].items():
        assert artifact["sha256_verified"] and not missing_scopes(artifact)
        rows = {}
        for group, operands in artifact["trace"].items():
            assert len(operands) == 2
            for row in operands:
                key = f"{group}/{row['operand_index']}"
                divisions = row["normalization_divisions"]
                branches = row["opaque_allgather_branches"]
                rows[key] = dict(
                    normalization_fields=[
                        dict(shape=n["shape"], opcode=n["opcode"], attributes=n["attributes"].split(", metadata=", 1)[0])
                        for n in divisions
                    ],
                    opaque_boundaries=[n["instruction"] for n in row["unresolved_boundaries"]],
                    auxiliary_allgather_branches=[
                        dict(
                            operand_index=b["operand_index"],
                            normalization_divisions=len(b["trace"]["normalization_divisions"]),
                            unresolved_boundaries=[n["instruction"] for n in b["trace"]["unresolved_boundaries"]],
                        )
                        for b in branches
                    ],
                )
        assert len(rows) == 14
        fields[name] = {key: row["normalization_fields"] for key, row in rows.items()}
        graphs[name] = dict(
            input_sha256=artifact["input_sha256"],
            extraction_rounds=artifact["extraction_rounds"],
            normalization_reached_positions=sum(bool(row["normalization_fields"]) for row in rows.values()),
            auxiliary_only_positions=sum(
                not row["normalization_fields"] and bool(row["auxiliary_allgather_branches"]) for row in rows.values()
            ),
            positions=rows,
        )
    comparisons = {}
    for solver in ("original", "matrix"):
        for variant in ("evaluate", "capture"):
            a, b = fields[solver + "_untouched"], fields[solver + "_" + variant]
            common = [k for k in a if a[k] and b[k]]
            comparisons[f"{solver}/untouched_vs_{variant}"] = dict(
                compared_positions=common,
                different_positions=[k for k in common if a[k] != b[k]],
                excluded_unresolved_positions=[k for k in a if k not in common],
            )
    return dict(
        graphs=graphs,
        comparisons=comparisons,
        comparison_scope=(
            "Ordered normalization result declarations, opcode and operation attributes before metadata. "
            "Operand values, caller bindings, backend configurations and runtime precision are not compared."
        ),
        limitations=(
            "ConcatBitcast traversal follows declared operands conservatively. AsyncCollectiveDone branches "
            "inspect one declared all-gather operand only; opaque result semantics remain unresolved. "
            "No missing record does not imply complete semantic provenance. Both capture reference gates failed."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.expected_sha256
    source = json.loads(raw)
    assert source["completed"]
    result = summarize(source)
    result["input_report_sha256"] = args.expected_sha256
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
