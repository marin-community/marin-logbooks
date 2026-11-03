# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Map captured full-update first-Gram reductions to validated isolated paths."""

import argparse
import hashlib
import json
from pathlib import Path

from compiler_operation_inventory import INSTRUCTION, OPERAND, TUPLE_INDEX


def differing_fields(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        return [
            p
            for k in sorted(left.keys() | right.keys())
            for p in differing_fields(left.get(k), right.get(k), path + "/" + k)
        ]
    return [] if left == right else [path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--isolated", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    completion = json.loads(args.completion.read_text())
    assert hashlib.sha256(raw).hexdigest() == completion["report_sha256"]
    assert len(raw) == completion["report_bytes"]
    source, isolated, previous = (
        json.loads(raw),
        json.loads(args.isolated.read_text()),
        json.loads(args.previous.read_text()),
    )
    assert source["completed"] and previous["input_report_sha256"] == completion["report_sha256"]
    result = dict(
        captured_report_sha256=completion["report_sha256"],
        captured_report_bytes=len(raw),
        isolated_record_sha256=hashlib.sha256(args.isolated.read_bytes()).hexdigest(),
        original_ir_manifest=source["manifest"]["artifacts"],
        graphs={},
        comparisons={},
    )
    for name in ("fused_original", "fused_matrix"):
        artifact = source["artifacts"][name]
        assert artifact["sha256_verified"]
        groups = artifact["polar_lineage"]["output_groups"]
        assert len(groups) == 7 and all(len(rows) == 15 for rows in groups.values())
        operations = {r["computation"] + "/" + r["name"]: r for r in artifact["selected_operations"]}
        entries = {INSTRUCTION.match(line)[1]: line for line in artifact["lineage_diagnostics"]["entry_instructions"]}
        mapped = {}
        for group in ("q_proj", "k_proj", "v_proj"):
            first = groups[group][0]
            assert first["iteration"] == 1 and first["product"] == "X_Xt"
            operation = first["operation"]
            matches = [
                (row, index)
                for row in artifact["lineage_diagnostics"]["entry_collective_inputs"]
                if row["opcode"] in ("all-reduce", "reduce-scatter")
                for index, ancestors in enumerate(row["polar_ancestors_by_operand"])
                if ancestors == [operation]
            ]
            assert len(matches) == 1
            collective, operand_index = matches[0]
            product = operations[operation]
            parents = [r for r in artifact["fusion_callers"] if r["calls"] == product["computation"]]
            assert len(parents) == 1
            parent = parents[0]
            assert parent["name"] in entries and collective["name"] in entries
            projection_line = entries[collective["operands"][operand_index]]
            projection = INSTRUCTION.match(projection_line)
            assert projection[3] == "get-tuple-element"
            assert OPERAND.findall(projection[4]) == [parent["name"]]
            assert int(TUPLE_INDEX.search(projection[5])[1]) == 0
            # The reduction itself is an entry instruction, not inside the product fusion.
            assert operations[parent["computation"] + "/" + collective["name"]]["opcode"] == "all-reduce"
            config = json.loads(collective["instruction"].split(", backend_config=", 1)[1])
            regression = [
                r
                for r in previous["graphs"][name]["entry_collectives_with_polar_ancestry"]
                if r["name"] == collective["name"]
            ]
            assert len(regression) == 1
            assert regression[0]["latest_polar_by_operand"][operand_index] == [group, 1, "X_Xt"]
            assert regression[0]["backend_config"] == config
            mapped[group] = dict(
                product=product,
                product_enclosing_fusion=parent,
                product_result_projection=projection_line,
                collective=collective,
                collective_operand_index=operand_index,
                collective_backend_config=config,
                collective_enclosing_fusion=None,
                scope_evidence=(
                    "Reduction is in the HLO entry computation; operand projects result0 of product fusion."
                ),
                previous_mapping_regression_passed=True,
            )
        assert mapped["q_proj"]["collective"]["name"] == mapped["k_proj"]["collective"]["name"]
        assert mapped["v_proj"]["collective"]["name"] != mapped["q_proj"]["collective"]["name"]
        result["graphs"][name] = mapped
    for group in ("q_proj", "k_proj", "v_proj"):
        left, right = (result["graphs"][name][group] for name in ("fused_original", "fused_matrix"))
        alg = left["collective_backend_config"]["collective_algorithm_config"]
        original_matrix_equal = alg == right["collective_backend_config"]["collective_algorithm_config"]
        assert original_matrix_equal
        controls = {}
        for layout in ("replicated", "direction_sharded"):
            graph = isolated["graphs"][f"{group}/collectives/reduce/{layout}"]
            configurations = [
                r["config"]["collective_algorithm_config"]
                for r in graph["backend_configs"]
                if r["config"].get("collective_algorithm_config")
            ]
            assert len(configurations) == 1
            controls[layout] = dict(equal=alg == configurations[0], configuration=configurations[0])
        assert controls["replicated"]["equal"] and not controls["direction_sharded"]["equal"]
        result["comparisons"][group] = dict(
            original_matrix_collective_algorithm_equal=original_matrix_equal,
            whole_collective_backend_different_fields=differing_fields(
                left["collective_backend_config"], right["collective_backend_config"]
            ),
            product_parent_backend_different_fields=differing_fields(
                left["product_enclosing_fusion"]["backend_config"], right["product_enclosing_fusion"]["backend_config"]
            ),
            isolated_algorithms=controls,
        )
    result["limitations"] = (
        "Current read verifies the captured regional report checksum; original IR checksum checks were performed "
        "by its recorded regional extraction, not repeated here. Mapping covers first-Gram q/k/v paths only. "
        "It preserves tuple-leaf operand correspondence but does not verify numerical identity inside producer fusions, "
        "runtime order or precision. Equal configurations do not establish equal execution "
        "or explain full-update errors."
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(mapped_paths=6, prior_mapping_regressions=6, comparisons=result["comparisons"]), indent=2))


if __name__ == "__main__":
    main()
