# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare output-aligned substitution records without inferring runtime causality."""

import argparse
import hashlib
import json
from pathlib import Path

from compare_capture_compiler import compare
from compare_full_update_collectives import differing_fields
from compiler_normalization_closure import missing_scopes
from compiler_operation_inventory import INSTRUCTION, backend_record


def summarize(source):
    result = compare(source)
    fields = {}
    for name, graph in source["artifacts"].items():
        closure = graph["normalization_closure"]
        assert not missing_scopes(closure)
        positions = {}
        for group, operands in closure["trace"].items():
            assert [r["operand_index"] for r in operands] == [0, 1]
            for row in operands:
                # Only unique reached divisions support field alignment at this position.
                unique = {n["computation"] + "/" + n["name"]: n for n in row["normalization_divisions"]}
                normalized = None
                if len(unique) == 1:
                    n = next(iter(unique.values()))
                    normalized = dict(
                        shape=n["shape"], opcode=n["opcode"], attributes=n["attributes"].split(", metadata=", 1)[0]
                    )
                positions[f'{group}/{row["operand_index"]}'] = dict(
                    normalization=normalized,
                    reached_divisions=len(unique),
                    opaque_boundaries=[r["instruction"] for r in row["unresolved_boundaries"]],
                    status="mapped" if normalized else "unresolved or ambiguous; no omission inferred",
                )
        collectives = {}
        for group, rows in graph["first_gram_collectives"].items():
            mapped = []
            for row in rows:
                match = INSTRUCTION.match(row["instruction"])
                config = row["instruction"].split(", backend_config=", 1)
                mapped.append(
                    dict(
                        opcode=match[3],
                        shape=match[2],
                        operand_index=row["operand_index"],
                        backend=backend_record(config[1]) if len(config) == 2 else None,
                    )
                )
            collectives[group] = mapped
        fields[name] = dict(normalization=positions, first_gram_collectives=collectives)
        memberships = {}
        for group, rows in graph["first_gram_collectives"].items():
            for row in rows:
                memberships.setdefault(row["name"], []).append(group)
        operations = {r["computation"] + "/" + r["name"]: r for r in graph["selected_operations"]}
        result["graphs"][name].update(
            entry_first_gram_collective_groups=list(memberships.values()),
            first_product_records={
                group: operations[chain[0]["operation"]]
                for group, chain in graph["polar_lineage"]["output_groups"].items()
            },
            external_input_bindings=graph["external_input_bindings"],
            normalization_positions=positions,
            first_gram_collectives=collectives,
            extraction_rounds=closure["extraction_rounds"],
            q_division_operand_provenance=graph["input_provenance"]["targets"]["q_proj/normalization"][
                "operand_provenance"
            ],
        )
    for pair, comparison in result["comparisons"].items():
        solver, names = pair.split("/")
        left, right = names.split("_vs_")
        a, b = fields[solver + "_" + left], fields[solver + "_" + right]
        common = [
            k
            for k in a["normalization"]
            if a["normalization"][k]["normalization"] and b["normalization"][k]["normalization"]
        ]
        comparison["normalization"] = dict(
            compared_positions=common,
            different_positions=[
                k for k in common if a["normalization"][k]["normalization"] != b["normalization"][k]["normalization"]
            ],
            excluded_positions=[k for k in a["normalization"] if k not in common],
        )
        comparison["collective_fields_different_groups"] = [
            g for g in a["first_gram_collectives"] if a["first_gram_collectives"][g] != b["first_gram_collectives"][g]
        ]
        comparison["unique_collective_different_fields"] = {
            g: differing_fields(a["first_gram_collectives"][g][0], b["first_gram_collectives"][g][0])
            for g in a["first_gram_collectives"]
            if len(a["first_gram_collectives"][g]) == len(b["first_gram_collectives"][g]) == 1
        }
    result["limitations"] += (
        " Both same-pair substitution gates failed. Normalization comparisons exclude operand values "
        "and caller bindings. Collective records include barrier IDs. Unresolved/absent mappings "
        "are not evidence of omitted source computations or dead code. No inference of runtime cause."
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.expected_sha256
    source = json.loads(raw)
    assert source["completed"] and len(source["artifacts"]) == 8
    result = summarize(source)
    result["input_report_sha256"] = args.expected_sha256
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
