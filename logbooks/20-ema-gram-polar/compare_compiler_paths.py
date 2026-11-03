# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare mapped static collective paths and matrix-kernel configurations."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
raw = args.report.read_bytes()
source = json.loads(raw)
assert source["completed"]
graphs = {}
products = {}
for name, artifact in source["artifacts"].items():
    assert artifact["sha256_verified"]
    positions = {
        row["operation"]: (group, row["iteration"], row["product"], index)
        for group, rows in artifact["polar_lineage"]["output_groups"].items()
        for index, row in enumerate(rows)
    }
    operations = {row["computation"] + "/" + row["name"]: row for row in artifact["selected_operations"]}
    callers = {}
    for row in artifact["fusion_callers"]:
        callers.setdefault(row["calls"], []).append(row)
    products[name] = {}
    for operation, position in positions.items():
        row = operations[operation]
        parents = callers[row["computation"]]
        assert len(parents) == 1, (name, operation)
        products[name]["/".join(map(str, position[:3]))] = {
            "attributes_before_metadata": row["attributes"].split(", metadata=", 1)[0],
            "operation_backend_config": row["backend_config"],
            "calling_fusion_backend_config": parents[0]["backend_config"],
        }
    collectives = []
    for row in artifact["lineage_diagnostics"]["entry_collective_inputs"]:
        operands = []
        for ancestors in row["polar_ancestors_by_operand"]:
            mapped = [positions[ancestor] for ancestor in ancestors]
            if not mapped:
                operands.append(None)
                continue
            assert len({position[0] for position in mapped}) == 1
            latest = max(mapped, key=lambda position: position[3])
            assert {position[3] for position in mapped} == set(range(latest[3] + 1))
            operands.append(latest[:3])
        if any(operands):
            config = row["instruction"].split(", backend_config=", 1)
            collectives.append(
                dict(
                    name=row["name"],
                    opcode=row["opcode"],
                    latest_polar_by_operand=operands,
                    backend_config=json.loads(config[1]) if len(config) == 2 else None,
                )
            )
    graphs[name] = dict(
        input_sha256_verified=True,
        mapped_products=len(positions),
        entry_collectives_with_polar_ancestry=collectives,
        first_query_product=products[name]["q_proj/1/X_Xt"],
    )


def differing_fields(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        return [
            field
            for key in sorted(left.keys() | right.keys())
            for field in differing_fields(left.get(key), right.get(key), path + "/" + key)
        ]
    return [] if left == right else [path]


comparisons = {}
for left, right in [("fused_original", "fused_matrix"), ("direction_original", "direction_matrix")]:
    assert products[left].keys() == products[right].keys()
    attribute_differences, operation_config_differences, parent_config_differences = [], [], {}
    for key, first in products[left].items():
        second = products[right][key]
        if first["attributes_before_metadata"] != second["attributes_before_metadata"]:
            attribute_differences.append(key)
        if first["operation_backend_config"] != second["operation_backend_config"]:
            operation_config_differences.append(key)
        fields = differing_fields(first["calling_fusion_backend_config"], second["calling_fusion_backend_config"])
        if fields:
            parent_config_differences[key] = fields
    comparisons[left + "_vs_" + right] = dict(
        compared_products=len(products[left]),
        attributes_before_metadata_different_positions=attribute_differences,
        operation_backend_config_different_positions=operation_config_differences,
        calling_fusion_backend_config_different_positions=parent_config_differences,
        calling_fusion_different_field_counts=dict(
            Counter(field for fields in parent_config_differences.values() for field in fields)
        ),
    )
report = dict(
    input_report_sha256=hashlib.sha256(raw).hexdigest(),
    graphs=graphs,
    comparisons=comparisons,
    limitations=(
        "Static compiler configuration and conservative dependencies, not runtime counts, reduction order, "
        "precision equivalence or numerical causality. Entry collectives exclude nested collectives. "
        "A null operand mapping means no mapped polar ancestry, not no numerical influence."
    ),
)
args.output.write_text(json.dumps(report, indent=2) + "\n")
