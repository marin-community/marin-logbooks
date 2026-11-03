# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Index saved trace coverage without making opaque outputs semantically transparent."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def identity(row, projection=True):
    result = [row["context"], row["computation"], row["name"]]
    if projection:
        result.append(row["tuple_path"])
    return encoded(result)


def declaration(row):
    return {k: row.get(k) for k in ("shape", "opcode", "operands", "parameter_index", "attributes")}


def traces(value, path=()):
    if isinstance(value, dict):
        if {"records", "edges", "boundaries", "root", "missing_records"} <= value.keys():
            yield path, value
        else:
            for key, child in value.items():
                yield from traces(child, (*path, str(key)))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from traces(child, (*path, str(i)))


def expansions(reports):
    """Explicit declared-input reports; input coverage is independent of output projection."""
    for endpoint in reports["boundary-inputs"]["endpoints"]:
        yield endpoint["endpoint"], endpoint["operand_traces"]
    for start in reports["start-producers"]["starts"]:
        yield start["start"], [{"operand_position": 0, "trace": start["declared_input_trace"]}]
    for group in reports["product-inputs"]["groups"].values():
        for endpoint in group.values():
            yield endpoint["endpoint"], endpoint["operands"]
    for endpoint in reports["polar-boundaries"]["groups"].values():
        yield endpoint["endpoint"], endpoint["operands"]
        if "matched_start" in endpoint:
            start = endpoint["matched_start"]
            yield start["record"], start["operands"]


def consolidate(reports, inventory):
    nodes, declarations = {}, {}
    uses, boundaries = defaultdict(list), defaultdict(list)
    edges = defaultdict(list)
    trace_counts, total_records = Counter(), 0
    input_positions = defaultdict(set)
    input_origins = defaultdict(list)

    def register(row):
        key, call = identity(row), identity(row, False)
        dec = declaration(row)
        if call in declarations:
            assert declarations[call] == dec, ("conflicting declaration", call)
        declarations[call] = dec
        nodes.setdefault(key, row)
        return key

    for report_name, report in reports.items():
        for path, trace in traces(report):
            assert not trace["missing_records"], (report_name, path, "missing records")
            trace_counts[report_name] += 1
            total_records += len(trace["records"])
            source = [report_name, list(path)]
            keys = [register(row) for row in trace["records"]]
            assert 0 <= trace["root"] < len(keys)
            for i, key in enumerate(keys):
                uses[key].append([source, i])
            for edge in trace["edges"]:
                left, right = keys[edge["source"]], keys[edge["target"]]
                position = edge["operand_index"]
                assert isinstance(position, int) and position >= 0
                edges[encoded([left, position, right])].append(source)
            for boundary in trace["boundaries"]:
                boundaries[keys[boundary["node"]]].append(dict(source=source, reason=boundary["reason"]))
    for row, operands in expansions(reports):
        register(row)
        call = identity(row, False)
        positions = [v["operand_position"] for v in operands]
        assert sorted(positions) == list(range(len(row["operands"]))), ("missing or duplicate position", call)
        for value in operands:
            root = value["trace"]["records"][value["trace"]["root"]]
            position = value["operand_position"]
            assert root["name"] == row["operands"][position], ("wrong operand root", call, position)
            assert root["computation"] == row["computation"] and root["context"] == row["context"]
            input_positions[call].add(position)
            input_origins[call].append(dict(position=position, root=identity(root)))
    outgoing = defaultdict(set)
    for key in edges:
        left, position, right = json.loads(key)
        outgoing[left].add(position)
    body_records = set()
    for field in ("producer_bodies", "first_gram_path_bodies"):
        for scope, lines in inventory[field].items():
            for line in lines:
                match = re.match(r"\s*(?:ROOT )?%([^ ]+) =", line)
                if match:
                    body_records.add((scope, match[1]))
    inventory_records = body_records | {(r["computation"], r["name"]) for r in inventory["selected_operations"]}
    entry_scopes = {r["computation"] for r in nodes.values() if not r["context"]}
    assert len(entry_scopes) == 1
    entry_scope = next(iter(entry_scopes))
    for line in inventory["lineage_diagnostics"]["entry_instructions"]:
        match = re.match(r"\s*(?:ROOT )?%([^ ]+) =", line)
        if match:
            inventory_records.add((entry_scope, match[1]))
    polar = {}
    for group, products in inventory["polar_lineage"]["output_groups"].items():
        for product in products:
            assert product["operation"] not in polar
            polar[product["operation"]] = dict(group=group, iteration=product["iteration"], product=product["product"])
    frontier_rows = []
    for key, occurrences in sorted(boundaries.items()):
        row = nodes[key]
        call = identity(row, False)
        target = re.search(r'custom_call_target="([^"]+)"', row.get("attributes", ""))
        kind = target[1] if target else row["opcode"]
        product = row["opcode"] in ("convolution", "dot")
        opaque = row["opcode"] in ("custom-call", "while", "call", "conditional") or any(
            "opaque" in x["reason"] for x in occurrences
        )
        explicit = sorted(input_positions[call])
        traversed = sorted(outgoing[key])
        complete = set(explicit if opaque else set(explicit) | set(traversed)) == set(range(len(row["operands"])))
        if opaque:
            category = "opaque_inputs_inspected" if complete else "opaque_inputs_not_fully_inspected"
        elif product:
            category = "product_inputs_inspected" if complete else "product_inputs_not_fully_inspected"
        elif row["opcode"] == "parameter" and not row["context"]:
            category = "source_parameter"
        else:
            category = "other_boundary"
        frontier_rows.append(
            dict(
                identity=json.loads(key),
                kind=kind,
                category=category,
                declaration=declaration(row),
                boundary_occurrences=len(occurrences),
                reports=sorted({x["source"][0] for x in occurrences}),
                explicit_declared_input_positions=explicit,
                saved_traversal_positions=traversed,
                inventory_body_record_present=(row["computation"], row["name"]) in body_records,
                inventory_scoped_record_present=(row["computation"], row["name"]) in inventory_records,
                declared_operand_record_coverage=[
                    dict(
                        operand_position=i,
                        operand_name=name,
                        same_context_record_present=encoded([row["context"], row["computation"], name]) in declarations,
                        inventory_scoped_record_present=(row["computation"], name) in inventory_records,
                    )
                    for i, name in enumerate(row["operands"])
                ],
                polar_role=polar.get(row["computation"] + "/" + row["name"]),
                output_remains_opaque=opaque,
            )
        )
    return dict(
        counts=dict(
            traces=dict(trace_counts),
            record_occurrences=total_records,
            unique_context_projection_records=len(nodes),
            unique_context_instruction_records=len(declarations),
            edge_occurrences=sum(map(len, edges.values())),
            unique_ordered_edges=len(edges),
            boundary_occurrences=sum(map(len, boundaries.values())),
            unique_boundary_context_projections=len(boundaries),
            frontier_categories=dict(Counter(x["category"] for x in frontier_rows)),
        ),
        frontiers=frontier_rows,
        records=nodes,
        record_uses=uses,
        edges=edges,
        boundary_uses=boundaries,
        explicit_input_origins=input_origins,
    )


def compact(graph):
    instructions = {}
    for frontier in graph["frontiers"]:
        key = encoded(frontier["identity"][:3])
        if key not in instructions:
            instructions[key] = {
                k: v
                for k, v in frontier.items()
                if k not in ("identity", "declaration", "boundary_occurrences", "reports", "saved_traversal_positions")
            }
            instructions[key].update(identity=frontier["identity"][:3], projections=[], occurrences=0, reports=[])
            if "not_fully" not in frontier["category"]:
                instructions[key].pop("declared_operand_record_coverage")
        row = instructions[key]
        assert row["category"] == frontier["category"], ("projection-dependent category", key)
        row["projections"].append(frontier["identity"][3])
        row["occurrences"] += frontier["boundary_occurrences"]
        row["reports"] = sorted(set(row["reports"]) | set(frontier["reports"]))
    counts = dict(graph["counts"])
    counts["unique_boundary_context_instructions"] = len(instructions)
    counts["instruction_frontier_categories"] = dict(Counter(x["category"] for x in instructions.values()))
    return dict(counts=counts, frontier_instructions=list(instructions.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    loaded = {}
    for name, entry in manifest["inputs"].items():
        raw = Path(entry["path"]).read_bytes()
        assert len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"], name
        loaded[name] = json.loads(raw)
        assert loaded[name]["completed"]
    inventory = loaded.pop("column-ir")
    graphs = {}
    for graph in ("matrix_untouched", "matrix_evaluate", "matrix_capture", "matrix_substitute"):
        hlo = inventory["manifest"]["artifacts"][graph]["sha256"]
        assert inventory["artifacts"][graph]["sha256_verified"]
        for name, report in loaded.items():
            artifact = report["artifacts"][graph]
            assert artifact["sha256_verified"] and artifact["input_sha256"] == hlo, (name, graph)
        graphs[graph] = consolidate(
            {name: report["artifacts"][graph]["trace"] for name, report in loaded.items()},
            inventory["artifacts"][graph],
        )
    common = dict(
        inputs=manifest,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=[
            "Joins stay within one HLO graph and retain caller context and tuple path; no cross-graph alignment.",
            "Explicit input coverage belongs to a declared instruction, not to semantics of any projected output.",
            "Opaque outputs remain opaque even when all declared operands have saved traces.",
            "Stored ordered edges preserve repeated operand positions and occurrences; union is not a runtime graph.",
            "Inventory record availability alone establishes neither complete bindings nor semantic provenance.",
            "Both numerical full28leaf substitution gates remain failed; no precision/order/value/cause claim.",
        ],
    )
    args.report.write_text(json.dumps(dict(**common, graphs=graphs), indent=2) + "\n")
    summary = {name: compact(value) for name, value in graphs.items()}
    args.summary.write_text(json.dumps(dict(**common, graphs=summary), indent=2) + "\n")


if __name__ == "__main__":
    main()
