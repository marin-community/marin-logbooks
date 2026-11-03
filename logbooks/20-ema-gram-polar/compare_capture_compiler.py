# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare output-mapped static product records in full-update capture graphs."""

import argparse
import hashlib
import json
from pathlib import Path

from inspect_product_paths import output_dependency_graph


def normalization_producers(artifact, operation):
    """Follow each immediate operand-producing fusion, stopping at its parameters."""
    result = []
    bodies = artifact["producer_bodies"]
    for index, producer in enumerate(operation["direct_operand_producers"]):
        callee = producer["calls"]
        if callee is None:
            result.append(dict(operand_index=index, producer=producer, mapped=False))
            continue
        assert callee in bodies
        lines = []
        for scope, body in bodies.items():
            lines.extend([("ENTRY " if scope == callee else "") + body[0], *body[1:]])
        graph = output_dependency_graph("\n".join(lines), ())
        divisions = [
            row
            for row in graph["records"]
            if row["opcode"] == "divide" and "/gram_direction/vmap()/div" in row["attributes"]
        ]
        result.append(
            dict(
                operand_index=index,
                producer=producer,
                mapped=bool(divisions),
                normalization_divisions=divisions,
                parameter_boundary="Immediate producer fusion parameters; caller values not resolved here",
            )
        )
    return result


def compare(report):
    products, graphs = {}, {}
    for name, graph in report["artifacts"].items():
        assert graph["sha256_verified"]
        operations = {row["computation"] + "/" + row["name"]: row for row in graph["selected_operations"]}
        callers = {}
        for row in graph["fusion_callers"]:
            callers.setdefault(row["calls"], []).append(row)
        mapped = {}
        for group, chain in graph["polar_lineage"]["output_groups"].items():
            for position in chain:
                row = operations[position["operation"]]
                parents = callers[row["computation"]]
                assert len(parents) == 1, (name, position["operation"])
                key = f"{group}/{position['iteration']}/{position['product']}"
                mapped[key] = dict(
                    shape=row["shape"],
                    operand_shapes=[p["shape"] for p in row["direct_operand_producers"]],
                    operand_opcodes=[p["opcode"] for p in row["direct_operand_producers"]],
                    operation_attributes=row["attributes"].split(", metadata=", 1)[0],
                    operation_backend=row["backend_config"],
                    enclosing_fusion_backend=parents[0]["backend_config"],
                )
        assert len(mapped) == 105
        products[name] = mapped
        graphs[name] = dict(
            mapped_products=105,
            output_roles=graph["output_roles"],
            normalization_records=len(graph["normalization"]["records"]),
            normalization_first_product_scope_matches=graph["normalization"]["first_product_scope_matches"],
            captured_producer_bodies=len(graph["producer_bodies"]),
            entry_collective_count=len(graph["lineage_diagnostics"]["entry_collective_inputs"]),
            first_product_normalization_operands={
                group: normalization_producers(graph, operations[chain[0]["operation"]])
                for group, chain in graph["polar_lineage"]["output_groups"].items()
            },
        )
    comparisons = {}
    for solver in ("original", "matrix"):
        for left, right in (("untouched", "evaluate"), ("untouched", "capture"), ("evaluate", "capture")):
            first, second = products[solver + "_" + left], products[solver + "_" + right]
            assert first.keys() == second.keys()
            fields = tuple(next(iter(first.values())))
            differences = {field: [k for k in first if first[k][field] != second[k][field]] for field in fields}
            comparisons[f"{solver}/{left}_vs_{right}"] = dict(
                matched_positions=105,
                differences=differences,
                affected_groups={
                    field: sorted({key.split("/")[0] for key in keys}) for field, keys in differences.items()
                },
            )
    return dict(
        graphs=graphs,
        comparisons=comparisons,
        limitations=(
            "Static mapped operation fields and immediate enclosing configurations, not runtime precision or cause. "
            "Operand shapes/opcodes omit upstream values and connections. Equal fields do not establish equal internal "
            "computations. Normalization scope associations are syntactic and may be empty. "
            "The underlying numerical capture reference gates both failed."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    source = json.loads(raw)
    assert source["completed"]
    result = compare(source)
    result["input_report_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
