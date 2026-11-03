# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize existing regional compiler artifacts without executing model kernels."""

import argparse
import gzip
import hashlib
import json
import os
import re
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

INSTRUCTION = re.compile(r"^\s*(?:ROOT\s+)?%([^ ]+) = (.*?) ([a-z][a-z0-9-]*)\(([^)]*)\)(.*)$")
COMPUTATION = re.compile(r"^(?:ENTRY )?%([^ ]+) .*\{$")
OP_NAME = re.compile(r'metadata=\{op_name="([^"]+)"')
CALLS = re.compile(r"calls=%([^, }]+)")
OPERAND = re.compile(r"%([a-zA-Z0-9_.-]+)")
STACK_FRAME = re.compile(r"stack_frame_id=(\d+)")
TUPLE_INDEX = re.compile(r"index=(\d+)")


@dataclass(frozen=True)
class ParameterLeaf:
    index: int
    path: tuple[int, ...]


def tuple_shapes(shape):
    if not shape.startswith("("):
        return None
    assert shape.endswith(")"), shape
    parts, start, depth = [], 1, 0
    for index, char in enumerate(shape[1:-1], 1):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(shape[start:index].strip())
            start = index + 1
    if shape[start:-1].strip():
        parts.append(shape[start:-1].strip())
    return parts


def shaped_dependencies(shape, leaf, path=()):
    parts = tuple_shapes(shape)
    if parts is None:
        return leaf(path)
    return tuple(shaped_dependencies(part, leaf, (*path, index)) for index, part in enumerate(parts))


def dependency_leaves(value):
    if isinstance(value, tuple):
        return frozenset().union(*(dependency_leaves(part) for part in value))
    return value


def substitute_dependencies(value, arguments):
    if isinstance(value, tuple):
        return tuple(substitute_dependencies(part, arguments) for part in value)
    dependencies = set()
    for token in value:
        if isinstance(token, ParameterLeaf):
            argument = arguments[token.index]
            for index in token.path:
                assert isinstance(argument, tuple), (token, argument)
                argument = argument[index]
            dependencies.update(dependency_leaves(argument))
        else:
            dependencies.add(token)
    return frozenset(dependencies)


def polar_lineage(lines, groups, diagnostics=None):
    """Follow fusion parameters and tuple leaves to associate polar operations with outputs.

    Other operations conservatively depend on every operand. Reject polar operations
    inside unsupported control flow or reused fusion instances instead of guessing.
    """
    if diagnostics is None:
        diagnostics = {}
    diagnostics["entry_instructions"] = []
    diagnostics["entry_collective_inputs"] = []
    summaries, nodes, events = {}, {}, {}
    scope, root, entry = None, None, None
    for line in lines:
        header = COMPUTATION.match(line.rstrip())
        if header:
            assert not nodes
            scope, root = header[1], None
            if line.startswith("ENTRY "):
                entry = scope
        if line.strip() == "}":
            assert root is not None, scope
            reachable = dependency_leaves(root)
            summaries[scope] = (root, {key: value for key, value in events.items() if key in reachable})
            nodes, events = {}, {}
            continue
        match = INSTRUCTION.match(line)
        if match is None:
            continue
        name, shape, opcode, operands, attributes = match.groups()
        if scope == entry:
            diagnostics["entry_instructions"].append(line.rstrip().split(", backend_config=", 1)[0])
        arguments = [nodes[key] for key in OPERAND.findall(operands)]
        deps = frozenset().union(*(dependency_leaves(arg) for arg in arguments))
        if scope == entry and opcode in ("all-reduce", "reduce-scatter", "all-gather"):
            diagnostics["entry_collective_inputs"].append(
                dict(
                    name=name,
                    opcode=opcode,
                    operands=OPERAND.findall(operands),
                    polar_ancestors_by_operand=[
                        sorted(token for token in dependency_leaves(arg) if isinstance(token, str)) for arg in arguments
                    ],
                    instruction=line.rstrip(),
                )
            )
        if opcode == "parameter":
            deps = shaped_dependencies(shape, lambda path, index=int(operands): frozenset({ParameterLeaf(index, path)}))
        elif opcode == "tuple":
            deps = tuple(arguments)
        elif opcode == "all-reduce" and len(arguments) > 1:
            # AllReduce operates independently on each tuple element.
            # https://openxla.org/xla/operation_semantics#allreduce
            assert len(tuple_shapes(shape)) == len(arguments), name
            for target in OPERAND.findall(attributes):
                if target in summaries:
                    assert not summaries[target][1], f"Unsupported polar reducer: {target}"
            deps = tuple(arguments)
        elif opcode == "get-tuple-element":
            assert isinstance(arguments[0], tuple), name
            deps = arguments[0][int(TUPLE_INDEX.search(attributes)[1])]
        elif opcode == "fusion":
            callee = CALLS.search(attributes)[1]
            callee_root, callee_events = summaries[callee]
            deps = substitute_dependencies(callee_root, arguments)
            for key, value in callee_events.items():
                assert key not in events, f"Repeated polar fusion instance: {key}"
                events[key] = substitute_dependencies(value, arguments)
        else:
            for target in OPERAND.findall(attributes):
                if target in summaries:
                    assert not summaries[target][1], f"Unsupported polar call: {opcode} -> {target}"
            deps = shaped_dependencies(shape, lambda _path, dependencies=deps: dependencies)
        metadata = OP_NAME.search(attributes)
        if (
            metadata
            and opcode in ("dot", "convolution")
            and (
                metadata[1].endswith("/gram_direction/vmap()/dot_general")
                or metadata[1].endswith("jit(polar)/vmap()/dot_general")
            )
        ):
            key = scope + "/" + name
            events[key] = dependency_leaves(deps)
            deps = dependency_leaves(deps) | {key}
        nodes[name] = deps
        if line.lstrip().startswith("ROOT "):
            root = deps
    assert not nodes and entry is not None
    outputs, events = summaries[entry]
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    diagnostics["output_polar_ancestors"] = [
        sorted(token for token in dependency_leaves(output) if isinstance(token, str)) for output in outputs
    ]
    assert len(outputs) >= len(groups)
    result, assigned = {}, set()
    for group, output in zip(groups, outputs, strict=False):
        keys = {token for token in dependency_leaves(output) if isinstance(token, str)}
        assert len(keys) == 15, (group, len(keys))
        assert not assigned.intersection(keys), f"Polar operations shared between output groups: {group}"
        ordered = sorted(keys, key=lambda key: len(events[key] & keys))
        preceding = set()
        mapped = []
        for index, key in enumerate(ordered):
            assert events[key] & keys == preceding, f"Polar dependency chain is not total: {group}/{key}"
            mapped.append(
                dict(operation=key, iteration=index // 3 + 1, product=("X_Xt", "scaled_A_A", "B_X")[index % 3])
            )
            preceding.add(key)
        result[group] = mapped
        assigned.update(keys)
    assert assigned == set(events), "Unassigned polar operations"
    return dict(
        output_groups=result,
        assigned_operations=len(assigned),
        method="Fusion parameter substitution and tuple-aware output ancestry; each group must form a total chain.",
        limitations=(
            "Static dependencies, not runtime ordering. Product roles follow the unchanged three-product "
            "quintic source after chain validation."
        ),
    )


def producer_bodies(lines, operations, extra_computations=()):
    """Collect immediate fusion bodies without loading the entire compiler module."""
    wanted = {producer["calls"] for op in operations for producer in op["direct_operand_producers"] if producer["calls"]}
    wanted.update(extra_computations)
    bodies, scope, body = {}, None, []
    for line in lines:
        header = COMPUTATION.match(line.rstrip())
        if header:
            scope, body = header[1], []
        if scope in wanted:
            # Runtime backend configuration is large; preserve the operation and metadata.
            body.append(line.rstrip().split(", backend_config=", 1)[0])
        if line.strip() == "}":
            if scope in wanted:
                bodies[scope] = body
            scope = None
    assert set(bodies) == wanted
    return bodies


def inventory(lines):
    """Retain direct producers of tagged polar and Gram-refresh matrix operations."""
    counts = Counter()
    operations = []
    fusion_callers, selected_scopes = [], set()
    nodes, selected = {}, []
    scope, total_lines = None, 0
    for number, line in enumerate(lines, 1):
        total_lines = number
        header = COMPUTATION.match(line.rstrip())
        if header:
            assert not nodes, "Unclosed computation"
            scope = header[1]
        if line.strip() == "}":
            if selected:
                selected_scopes.add(scope)
            for node in selected:
                node["direct_operand_producers"] = [
                    {k: nodes[name][k] for k in ("name", "shape", "opcode", "calls")} for name in node["operands"]
                ]
                operations.append(node)
            nodes, selected = {}, []
            continue
        match = INSTRUCTION.match(line)
        if match is None:
            continue
        name, shape, opcode, operands, attributes = match.groups()
        counts["instructions"] += 1
        counts["opcode/" + opcode] += 1
        if opcode == "convert" and shape.startswith("bf16["):
            counts["converts_to_bf16"] += 1
        call = CALLS.search(attributes)
        node = dict(name=name, shape=shape, opcode=opcode, calls=call[1] if call else None)
        nodes[name] = node
        config = attributes.split(", backend_config=", 1)
        captured_attributes = dict(
            attributes=config[0], backend_config=json.loads(config[1]) if len(config) == 2 else None
        )
        if opcode == "fusion" and call and call[1] in selected_scopes:
            fusion_callers.append(
                dict(**node, computation=scope, operands=OPERAND.findall(operands), **captured_attributes)
            )
        metadata = OP_NAME.search(attributes)
        if metadata is None:
            continue
        op_name = metadata[1]
        polar = op_name.endswith("/gram_direction/vmap()/dot_general") or op_name.endswith(
            "jit(polar)/vmap()/dot_general"
        )
        refresh = op_name.endswith("/update_gram_state/dot_general")
        if (polar or refresh) and opcode in ("dot", "convolution", "all-reduce", "reduce-scatter", "all-gather"):
            frame = STACK_FRAME.search(attributes)
            selected.append(
                dict(
                    **node,
                    line=number,
                    computation=scope,
                    family="polar" if polar else "refresh",
                    op_name=op_name,
                    stack_frame_id=int(frame[1]) if frame else None,
                    operands=OPERAND.findall(operands),
                    **captured_attributes,
                )
            )
    assert not nodes, "Incomplete compiler text"
    return dict(
        lines=total_lines, static_counts=dict(counts), selected_operations=operations, fusion_callers=fusion_callers
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    directory = StoragePath(args.output_prefix) / ("ir-inventory-" + uuid.uuid4().hex[:12])
    directory.mkdirs()
    output = directory / "report.json"
    report = dict(
        source_commit=os.environ.get("GIT_COMMIT"),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        manifest=manifest,
        completed=False,
        artifacts={},
        limitations="Static text inventory, not runtime counts or causal proof. No numerical kernels or training.",
    )
    with tempfile.TemporaryDirectory() as directory:
        for name, artifact in manifest["artifacts"].items():
            compressed = Path(directory) / (name + ".txt.gz")
            digest = hashlib.sha256()
            size = 0
            with (
                StoragePath(artifact["uri"]).open("rb", requests_timeout=60) as source,
                compressed.open("wb") as target,
            ):
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    target.write(chunk)
            assert size == artifact["compressed_bytes"] and digest.hexdigest() == artifact["sha256"], name
            with gzip.open(compressed, "rt") as lines:
                report["artifacts"][name] = inventory(lines)
            report["artifacts"][name]["sha256_verified"] = True
            if manifest.get("analysis") == "selected-computations":
                bodies = report["artifacts"][name]["selected_computations"] = {}
                pending = set(artifact["computation_names"])
                while pending:
                    with gzip.open(compressed, "rt") as lines:
                        captured = producer_bodies(lines, [], pending)
                    assert pending <= captured.keys(), sorted(pending - captured.keys())
                    bodies.update(captured)
                    pending = {
                        match[1] for body in captured.values() for line in body for match in CALLS.finditer(line)
                    } - bodies.keys()
                    report["active_stage"] = name + "/selected_computations"
                    output.write_text(json.dumps(report, indent=2), requests_timeout=60)
            if manifest.get("analysis") == "polar-lineage":
                report["active_stage"] = name + "/polar_lineage"
                output.write_text(json.dumps(report, indent=2), requests_timeout=60)
                diagnostics = report["artifacts"][name]["lineage_diagnostics"] = {}
                try:
                    with gzip.open(compressed, "rt") as lines:
                        report["artifacts"][name]["polar_lineage"] = polar_lineage(
                            lines, manifest["groups"], diagnostics
                        )
                except AssertionError:
                    output.write_text(json.dumps(report, indent=2), requests_timeout=60)
                    raise
                report["active_stage"] = name + "/producer_bodies"
                output.write_text(json.dumps(report, indent=2), requests_timeout=60)
                with gzip.open(compressed, "rt") as lines:
                    report["artifacts"][name]["producer_bodies"] = producer_bodies(
                        lines, report["artifacts"][name]["selected_operations"]
                    )
            output.write_text(json.dumps(report, indent=2), requests_timeout=60)
            print(
                json.dumps(
                    dict(artifact=name, selected_operations=len(report["artifacts"][name]["selected_operations"]))
                ),
                flush=True,
            )
            compressed.unlink()
    report["completed"] = True
    report["active_stage"] = "completed"
    output.write_text(json.dumps(report, indent=2), requests_timeout=60)
    print(json.dumps(dict(completed=True, output=str(output))), flush=True)


if __name__ == "__main__":
    main()
