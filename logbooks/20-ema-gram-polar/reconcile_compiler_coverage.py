# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile exact input-root coverage across saved reports, retaining opaque outputs."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from consolidate_compiler_records import declaration, encoded, identity, traces


def reconcile(reports, inventory):
    records, declarations = {}, {}
    roots, edges, boundaries, uses = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    counts = Counter()
    trace_counts = Counter()
    for label, report in reports.items():
        for path, value in traces(report):
            assert not value["missing_records"], (label, path, "requested missing records")
            rows = value["records"]
            assert 0 <= value["root"] < len(rows)
            origin = dict(report=label, path=list(path))
            keys = []
            for i, row in enumerate(rows):
                key, call = identity(row), identity(row, False)
                dec = declaration(row)
                assert call not in declarations or declarations[call] == dec, ("conflicting declaration", call)
                declarations[call] = dec
                records.setdefault(key, row)
                uses[key].append(dict(**origin, node=i))
                keys.append(key)
            roots[keys[value["root"]]].append(origin)
            edge_positions = set()
            for edge in value["edges"]:
                a, b, pos = edge["source"], edge["target"], edge["operand_index"]
                assert 0 <= a < len(keys) and 0 <= b < len(keys) and isinstance(pos, int) and pos >= 0
                assert (a, pos) not in edge_positions, "Duplicate outgoing operand position"
                edge_positions.add((a, pos))
                edges[encoded([keys[a], pos, keys[b]])].append(origin)
            for boundary in value["boundaries"]:
                assert 0 <= boundary["node"] < len(keys)
                boundaries[keys[boundary["node"]]].append(dict(**origin, reason=boundary["reason"]))
            trace_counts[label] += 1
            counts.update(record_occurrences=len(rows), edge_occurrences=len(value["edges"]))
    polar = {
        row["operation"]: dict(group=group, iteration=row["iteration"], product=row["product"])
        for group, chain in inventory["polar_lineage"]["output_groups"].items()
        for row in chain
    }
    instructions = {}
    for key, origins in sorted(boundaries.items()):
        row = records[key]
        call = identity(row, False)
        match = re.search(r'custom_call_target="([^"]+)"', row.get("attributes", ""))
        opaque = row["opcode"] in ("custom-call", "while", "call", "conditional") or any(
            "opaque" in origin["reason"] for origin in origins
        )
        if call not in instructions:
            positions = []
            for position, operand in enumerate(row["operands"]):
                root_key = identity(dict(row, name=operand, tuple_path=[]))
                positions.append(
                    dict(
                        operand_position=position,
                        operand_name=operand,
                        input_root_identity=json.loads(root_key),
                        complete_input_roots=roots.get(root_key, []),
                        record_present=root_key in records,
                    )
                )
            category = (
                "source_parameter"
                if row["opcode"] == "parameter" and not row["context"]
                else "opaque" if opaque else "product" if row["opcode"] in ("dot", "convolution") else "other"
            )
            instructions[call] = dict(
                identity=json.loads(call),
                declaration=declaration(row),
                kind=match[1] if match else row["opcode"],
                category=category,
                output_remains_opaque=opaque,
                polar_role=polar.get(row["computation"] + "/" + row["name"]),
                projections=[],
                boundary_origins=[],
                declared_input_positions=positions,
            )
        target = instructions[call]
        assert target["output_remains_opaque"] == opaque
        target["projections"].append(row["tuple_path"])
        target["boundary_origins"].extend(origins)
    rows = list(instructions.values())
    for row in rows:
        row["uncovered_positions"] = [
            p["operand_position"] for p in row["declared_input_positions"] if not p["complete_input_roots"]
        ]
    counts.update(
        unique_context_projection_records=len(records),
        unique_context_instructions=len(declarations),
        unique_ordered_edges=len(edges),
        boundary_occurrences=sum(map(len, boundaries.values())),
        unique_boundary_context_projections=len(boundaries),
        unique_boundary_context_instructions=len(rows),
        uncovered_instructions=sum(bool(row["uncovered_positions"]) for row in rows),
        uncovered_positions=sum(len(row["uncovered_positions"]) for row in rows),
    )
    return dict(
        counts=dict(counts),
        traces=dict(trace_counts),
        instructions=rows,
        records=records,
        roots=roots,
        ordered_edges=edges,
        record_uses=uses,
        boundary_uses=boundaries,
    )


def compact(graph):
    gaps = []
    for row in graph["instructions"]:
        if not row["uncovered_positions"]:
            continue
        gaps.append(
            dict(
                **{k: v for k, v in row.items() if k not in ("boundary_origins", "declared_input_positions")},
                output_dependencies=sorted(
                    {encoded(dict(report=o["report"], path=o["path"])) for o in row["boundary_origins"]}
                ),
                declared_input_positions=[
                    dict(
                        **{k: v for k, v in p.items() if k != "complete_input_roots"},
                        complete_input_root_count=len(p["complete_input_roots"]),
                    )
                    for p in row["declared_input_positions"]
                ],
            )
        )
    return dict(
        counts=graph["counts"],
        traces=graph["traces"],
        frontier_kinds=dict(Counter(row["kind"] for row in graph["instructions"])),
        gap_kinds=dict(Counter(row["kind"] for row in gaps)),
        gaps=gaps,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    loaded = {}
    for label, entry in manifest["inputs"].items():
        raw = Path(entry["path"]).read_bytes()
        assert len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"]
        loaded[label] = json.loads(raw)
        assert loaded[label]["completed"]
    inventory = loaded.pop("inventory")
    graphs = {}
    for name in manifest["graphs"]:
        sha = inventory["manifest"]["artifacts"][name]["sha256"]
        assert inventory["artifacts"][name]["sha256_verified"]
        saved = {}
        for label, report in loaded.items():
            if name not in report["artifacts"]:
                assert label == "capture-initializer-concat" and name in ("matrix_untouched", "matrix_evaluate")
                continue
            graph = report["artifacts"][name]
            assert graph["sha256_verified"] and graph["input_sha256"] == sha
            saved[label] = graph["trace"]
        graphs[name] = reconcile(saved, inventory["artifacts"][name])
    common = dict(
        manifest=manifest,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=[
            "Exact graph/caller/computation/instruction/tuple-path joins; no cross-graph alignment.",
            "Coverage requires a complete saved trace root at each declared operand and empty tuple projection.",
            "Ordered edges, repeated positions, output dependencies and source scopes remain recorded.",
            "Covered inputs do not resolve opaque outputs. Deliberate gaps are not requested missing records.",
            "The union of saved edges is not a runtime graph. Counts do not establish cross-graph identity.",
            "No new HLO parse, job, accelerator, numerical kernels or W&B; prior regional HLO identity joins only.",
            "Both full28leaf numerical gates remain failed; no values/precision/order/identity/cause claim.",
        ],
    )
    args.report.write_text(json.dumps(dict(**common, graphs=graphs), separators=(",", ":")) + "\n")
    args.summary.write_text(
        json.dumps(dict(**common, graphs={n: compact(g) for n, g in graphs.items()}), separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    main()
