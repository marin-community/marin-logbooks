# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Follow a selected output through small saved HLO graphs, including fusion bindings."""

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

from compiler_operation_inventory import CALLS, COMPUTATION, INSTRUCTION, OPERAND, TUPLE_INDEX
from rigging.filesystem.storage_path import StoragePath


def output_dependency_graph(text, output_path):
    scopes, roots, entry = {}, {}, None
    for line in text.splitlines():
        header = COMPUTATION.match(line)
        if header:
            scope = header[1]
            scopes[scope] = {}
            if line.startswith("ENTRY "):
                entry = scope
        match = INSTRUCTION.match(line)
        if match:
            name, shape, opcode, operands, attrs = match.groups()
            scopes[scope][name] = dict(
                name=name,
                shape=shape,
                opcode=opcode,
                operands=OPERAND.findall(operands),
                parameter_index=int(operands) if opcode == "parameter" else None,
                attributes=attrs,
                instruction=line.strip(),
            )
            if line.lstrip().startswith("ROOT "):
                roots[scope] = name
    assert entry is not None
    records, visited, edges, stack = [], set(), {}, []

    def walk(scope, name, bindings, context, path=()):
        identity = json.dumps([context, scope, name, path])
        if stack:
            edges[stack[-1]].add(identity)
        if identity in visited:
            return
        edges[identity] = set()
        stack.append(identity)
        visit(scope, name, bindings, context, path, identity)
        stack.pop()

    def visit(scope, name, bindings, context, path, identity):
        if identity in visited:
            return
        visited.add(identity)
        node = scopes[scope][name]
        records.append(
            dict(identity=identity, context=list(context), computation=scope, selected_tuple_path=list(path), **node)
        )
        op, operands = node["opcode"], node["operands"]
        if op == "parameter":
            if bindings:
                parent_scope, parent_name, parent_bindings, parent_context = bindings[node["parameter_index"]]
                walk(parent_scope, parent_name, parent_bindings, parent_context, path)
            return
        if op == "get-tuple-element":
            walk(scope, operands[0], bindings, context, (int(TUPLE_INDEX.search(node["attributes"])[1]), *path))
            return
        if op == "tuple":
            assert path, "A tuple output must select a leaf"
            walk(scope, operands[path[0]], bindings, context, path[1:])
            return
        if op == "copy-done":
            walk(scope, operands[0], bindings, context, (0, *path))
            return
        if op == "copy-start":
            assert path and path[0] in (0, 1) and len(operands) == 1
            walk(scope, operands[0], bindings, context, path[1:])
            return
        if op == "fusion":
            callee = CALLS.search(node["attributes"])[1]
            arguments = [(scope, operand, bindings, context) for operand in operands]
            walk(callee, roots[callee], arguments, (*context, scope + "/" + name), path)
            return
        if op == "all-reduce" and len(operands) > 1:
            assert path
            walk(scope, operands[path[0]], bindings, context, path[1:])
            return
        assert not path, (scope, name, path)
        if op == "custom-call":
            assert 'custom_call_target="ConcatBitcast"' in node["attributes"], (scope, name, node["attributes"])
            records[-1]["dependency_policy"] = "Opaque ConcatBitcast: conservatively follow all declared operands"
        assert op not in ("while", "call", "conditional"), (scope, name, op)
        for operand in operands:
            walk(scope, operand, bindings, context)

    walk(entry, roots[entry], [], (), output_path)
    return dict(
        entry=entry,
        output_path=list(output_path),
        dependencies={key: sorted(value) for key, value in edges.items()},
        records=records,
    )


def output_ancestors(text, output_path):
    graph = output_dependency_graph(text, output_path)
    records = graph["records"]
    products = [row for row in records if row["opcode"] in ("dot", "convolution")]
    assert len(products) == 1, len(products)
    return dict(
        entry=graph["entry"],
        output_path=graph["output_path"],
        matrix_product=products[0],
        collectives=[row for row in records if row["opcode"] in ("all-reduce", "reduce-scatter", "all-gather")],
        declared_bf16_nodes=[row for row in records if row["shape"].startswith("bf16[")],
        records=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    completion = json.loads(args.completion.read_text())
    manifest = completion["compiler_artifacts"]["manifest"]
    report = dict(
        source_completion_sha256=hashlib.sha256(args.completion.read_bytes()).hexdigest(),
        graphs={},
        limitations="Static output ancestry, not runtime order, arithmetic precision or numerical causality.",
    )
    for group in ("q_proj", "k_proj", "v_proj"):
        for kind in ("products", "fixed_product/A"):
            for layout in ("replicated", "direction_sharded"):
                key = f"{group}/{kind}/{layout}"
                artifact = manifest[key]["optimized_hlo"]
                data = StoragePath(artifact["uri"]).read_bytes(requests_timeout=60)
                assert (
                    len(data) == artifact["compressed_bytes"] and hashlib.sha256(data).hexdigest() == artifact["sha256"]
                )
                text = gzip.decompress(data).decode()
                if kind == "products":
                    stable = manifest[key]["stablehlo"]
                    stable_data = StoragePath(stable["uri"]).read_bytes(requests_timeout=60)
                    assert len(stable_data) == stable["compressed_bytes"]
                    assert hashlib.sha256(stable_data).hexdigest() == stable["sha256"]
                    fields = re.findall(
                        r'jax.result_info = "result\[\x27([^\x27]+)\x27\]"', gzip.decompress(stable_data).decode()
                    )
                    assert fields == ["A", "B", "C", "D", "X1", "final", "norm", "normalized_input"], fields
                    path = (fields.index("A"),)
                else:
                    path = ()
                graph = output_ancestors(text, path)
                graph.update(artifact=artifact, sha256_verified=True)
                if kind == "products":
                    graph["output_mapping"] = dict(
                        stablehlo_artifact=stable, stablehlo_sha256_verified=True, fields=fields
                    )
                report["graphs"][key] = graph
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
