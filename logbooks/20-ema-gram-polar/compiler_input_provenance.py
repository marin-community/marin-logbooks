# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve selected compiler operands to conservative entry-parameter dependencies."""

import argparse
import gzip
import hashlib
import json
import os
import re
import tempfile
import uuid
from pathlib import Path

from compiler_operation_inventory import (
    CALLS,
    COMPUTATION,
    INSTRUCTION,
    OP_NAME,
    OPERAND,
    TUPLE_INDEX,
    ParameterLeaf,
    dependency_leaves,
    shaped_dependencies,
    substitute_dependencies,
    tuple_shapes,
)
from rigging.filesystem.storage_path import StoragePath


def provenance(lines, targets):
    by_operation = {v["operation"]: k for k, v in targets.items()}
    assert len(by_operation) == len(targets)
    summaries, nodes, events, captured, parameters = {}, {}, {}, {}, {}
    scope = entry = root = None
    for line in lines:
        header = COMPUTATION.match(line.rstrip())
        if header:
            assert not nodes
            scope, root = header[1], None
            if line.startswith("ENTRY "):
                entry = scope
        if line.strip() == "}":
            assert root is not None
            reachable = dependency_leaves(root)
            summaries[scope] = (root, {k: v for k, v in events.items() if "event:" + k in reachable})
            nodes, events = {}, {}
            continue
        match = INSTRUCTION.match(line)
        if match is None:
            continue
        name, shape, opcode, operand_text, attrs = match.groups()
        operand_names = OPERAND.findall(operand_text)
        arguments = [nodes[n] for n in operand_names]
        deps = frozenset().union(*(dependency_leaves(x) for x in arguments))
        if opcode == "parameter":
            index = int(operand_text)
            deps = shaped_dependencies(shape, lambda path, index=index: frozenset({ParameterLeaf(index, path)}))
            if scope == entry:
                metadata = OP_NAME.search(attrs)
                parameters[index] = dict(shape=shape, source_label=metadata[1] if metadata else None, name=name)
        elif opcode == "tuple":
            deps = tuple(arguments)
        elif opcode == "get-tuple-element":
            assert isinstance(arguments[0], tuple)
            deps = arguments[0][int(TUPLE_INDEX.search(attrs)[1])]
        elif opcode == "all-reduce" and len(arguments) > 1:
            assert len(tuple_shapes(shape)) == len(arguments)
            deps = tuple(arguments)
        elif opcode == "fusion":
            callee = CALLS.search(attrs)[1]
            callee_root, callee_events = summaries[callee]
            deps = substitute_dependencies(callee_root, arguments)
            for key, value in callee_events.items():
                assert key not in events, f"Reused selected fusion: {key}"
                events[key] = substitute_dependencies(value, arguments)
        else:
            for target in OPERAND.findall(attrs):
                if target in summaries:
                    assert not summaries[target][1], f"Selected target in unsupported call: {opcode}"
            if opcode in ("custom-call", "while", "conditional", "call", "async-start", "async-done"):
                target = re.search(r'custom_call_target="([^"]+)"', attrs)
                deps |= {"opaque:" + opcode + (":" + target[1] if target else "")}
            deps = shaped_dependencies(shape, lambda _path, dependencies=deps: dependencies)
        identity = scope + "/" + name
        if identity in by_operation:
            label = by_operation[identity]
            assert len(arguments) == 2 and opcode in ("divide", "dot", "convolution")
            assert label not in captured
            captured[label] = dict(
                operation=identity,
                opcode=opcode,
                shape=shape,
                operands=operand_names,
                instruction=line.rstrip().split(", backend_config=", 1)[0],
            )
            events[label] = tuple(arguments)
            deps = dependency_leaves(deps) | {"event:" + label}
        nodes[name] = deps
        if line.lstrip().startswith("ROOT "):
            root = deps
    assert not nodes and entry in summaries
    output_events = summaries[entry][1]
    assert output_events.keys() == targets.keys(), (output_events.keys(), targets.keys())
    result = {}
    for label, arguments in output_events.items():
        rows = []
        for arg in arguments:
            leaves = dependency_leaves(arg)
            refs = sorted((v for v in leaves if isinstance(v, ParameterLeaf)), key=lambda v: (v.index, v.path))
            rows.append(
                dict(
                    entry_parameters=[dict(index=v.index, tuple_path=list(v.path), **parameters[v.index]) for v in refs],
                    conservative_operations=sorted(x for x in leaves if isinstance(x, str) and x.startswith("opaque:")),
                )
            )
        result[label] = dict(**captured[label], role=targets[label]["role"], operand_provenance=rows)
    return dict(
        entry_computation=entry,
        entry_parameters=parameters,
        targets=result,
        method=(
            "Fusion-parameter substitution; tuple-leaf projections; combined all-reduce operands kept independent. "
            "Other operations conservatively use all operands."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    destination = StoragePath(args.output_prefix) / ("input-provenance-" + uuid.uuid4().hex[:12])
    destination.mkdirs()
    result = dict(
        source_commit=os.environ.get("GIT_COMMIT"),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest=manifest,
        completed=False,
        artifacts={},
    )
    with tempfile.TemporaryDirectory() as directory:
        for name, artifact in manifest["artifacts"].items():
            path = Path(directory) / (name + ".gz")
            digest, count = hashlib.sha256(), 0
            with StoragePath(artifact["uri"]).open("rb", requests_timeout=60) as source, path.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    count += len(chunk)
                    output.write(chunk)
            assert count == artifact["compressed_bytes"] and digest.hexdigest() == artifact["sha256"]
            with gzip.open(path, "rt") as lines:
                result["artifacts"][name] = provenance(lines, artifact["targets"])
            result["artifacts"][name]["input_sha256_verified"] = True
            path.unlink()
            (destination / "report.json").write_text(json.dumps(result, indent=2), requests_timeout=60)
            print(
                json.dumps(dict(artifact=name, targets=len(artifact["targets"]), finite_scope="static text only")),
                flush=True,
            )
    result["completed"] = True
    result["limitations"] = (
        "Conservative static dependencies, not minimal semantic inputs, numerical identity, "
        "runtime precision or causal proof. No kernels or training."
    )
    (destination / "report.json").write_text(json.dumps(result, indent=2), requests_timeout=60)
    print(json.dumps(dict(completed=True, report=str(destination / "report.json"))), flush=True)


if __name__ == "__main__":
    main()
