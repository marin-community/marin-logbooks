# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Align saved polar products by validated final-output dependencies."""

import argparse
import gzip
import hashlib
import json
import re
from functools import cache
from pathlib import Path

from inspect_product_paths import output_dependency_graph
from rigging.filesystem.storage_path import StoragePath


def aligned_products(text, path, kind):
    graph = output_dependency_graph(text, path)
    records = {row["identity"]: row for row in graph["records"]}
    edges = graph["dependencies"]

    @cache
    def ancestors(key):
        return frozenset(edges[key]).union(*(ancestors(child) for child in edges[key]))

    products = {key for key, row in records.items() if row["opcode"] in ("dot", "convolution")}
    expected = 14 if kind == "continuation" else 15
    assert len(products) == expected, (kind, len(products))
    ordered = sorted(products, key=lambda key: len(ancestors(key) & products))
    preceding, result = set(), []
    for index, key in enumerate(ordered):
        assert ancestors(key) & products == preceding, (kind, index, "non-total dependency chain")
        row = records[key]
        assert re.search(
            r'op_name="jit\((polar|polar_products|polar_continuation)\)/vmap\(\)/dot_general"', row["attributes"]
        )
        offset = index + (kind == "continuation")
        role = ("A", "C", "D")[offset % 3]
        iteration = offset // 3 + 1
        labels = re.search(r"dim_labels=([^, ]+)", row["attributes"])
        assert labels and labels[1] == ("0bf_0oi->0bf" if role == "A" else "0bf_0io->0bf")
        boundary = set()

        def frontier(node, boundary):
            if node in boundary:
                return
            boundary.add(node)
            if node not in products:
                for child in edges[node]:
                    frontier(child, boundary)

        for child in edges[key]:
            frontier(child, boundary)
        parents = []
        for context in row["context"]:
            scope, name = context.split("/", 1)
            matches = [r for r in records.values() if r["computation"] == scope and r["name"] == name]
            assert len(matches) == 1, (context, len(matches))
            parents.append(matches[0])
        operand_by_name = {records[child]["name"]: records[child] for child in edges[key]}
        # Keep repeated operands (X @ X.T may lower to convolution(X, X)).
        operand_records = [operand_by_name[name] for name in row["operands"]]
        result.append(
            dict(
                iteration=iteration,
                product=role,
                operation=row,
                operands=operand_records,
                calling_fusions=parents,
                upstream_product_count=len(preceding),
                operand_frontier=[records[node] for node in sorted(boundary)],
                collective_ancestors=[
                    records[node]
                    for node in sorted(ancestors(key))
                    if records[node]["opcode"] in ("all-reduce", "all-gather", "reduce-scatter")
                ],
            )
        )
        preceding.add(key)
    return dict(products=result, record_count=len(records), validated_total_chain=True)


def backend_config(row):
    parts = row["attributes"].split(", backend_config=", 1)
    return json.loads(parts[1]) if len(parts) == 2 else None


def compact_report(report):
    result = dict(graphs={}, comparisons={}, limitations=report["limitations"])
    for key, graph in report["graphs"].items():
        products = []
        for product in graph["products"]:
            products.append(
                dict(
                    iteration=product["iteration"],
                    product=product["product"],
                    operation=product["operation"]["computation"] + "/" + product["operation"]["name"],
                    operand_shapes=[row["shape"] for row in product["operands"]],
                    attributes_before_metadata=product["operation"]["attributes"].split(", metadata=", 1)[0],
                    operation_backend_config=backend_config(product["operation"]),
                    calling_fusion_backend_configs=[backend_config(row) for row in product["calling_fusions"]],
                    frontier_converts=[
                        dict(shape=row["shape"], instruction=row["instruction"])
                        for row in product["operand_frontier"]
                        if row["opcode"] == "convert"
                    ],
                    frontier_collectives=[
                        dict(shape=row["shape"], instruction=row["instruction"])
                        for row in product["operand_frontier"]
                        if row["opcode"] in ("all-reduce", "all-gather", "reduce-scatter")
                    ],
                )
            )
        result["graphs"][key] = dict(
            products=products,
            artifacts=graph["artifacts"],
            first_product_normalization_records=[
                row
                for row in graph["products"][0]["operand_frontier"]
                if "jit(norm)" in row["attributes"] or "/vmap()/div" in row["attributes"]
            ],
            first_product_calling_fusions=graph["products"][0]["calling_fusions"],
            sha256_size_and_local_bytes_verified=graph["sha256_size_and_local_bytes_verified"],
            validated_total_chain=graph["validated_total_chain"],
            output_path=graph["output_path"],
        )
    fields = (
        "operand_shapes",
        "attributes_before_metadata",
        "operation_backend_config",
        "calling_fusion_backend_configs",
    )
    for group in ("q_proj", "k_proj", "v_proj"):
        for layout in ("replicated", "direction_sharded"):
            for left, right in (
                ("identical_inner", "products"),
                ("identical_inner", "continuation"),
                ("products", "continuation"),
            ):
                first = {
                    (p["iteration"], p["product"]): p for p in result["graphs"][f"{group}/{left}/{layout}"]["products"]
                }
                second = {
                    (p["iteration"], p["product"]): p for p in result["graphs"][f"{group}/{right}/{layout}"]["products"]
                }
                matched = sorted(first.keys() & second.keys())
                result["comparisons"][f"{group}/{layout}/{left}_vs_{right}"] = dict(
                    matched_products=len(matched),
                    different_positions={
                        field: [f"{i}/{role}" for i, role in matched if first[i, role][field] != second[i, role][field]]
                        for field in fields
                    },
                )
    configs = {}

    def config_reference(config):
        if config is None:
            return None
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        configs[digest] = config
        return digest

    for graph in result["graphs"].values():
        for product in graph["products"]:
            product["operation_backend_config_sha256"] = config_reference(product.pop("operation_backend_config"))
            product["calling_fusion_backend_config_sha256s"] = [
                config_reference(config) for config in product.pop("calling_fusion_backend_configs")
            ]
            for field in ("frontier_converts", "frontier_collectives"):
                for row in product[field]:
                    parts = row["instruction"].split(", backend_config=", 1)
                    row["instruction"] = parts[0]
                    row["backend_config_sha256"] = config_reference(json.loads(parts[1]) if len(parts) == 2 else None)
    result["backend_configs_by_sha256"] = configs
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.completion.read_text())["compiler_artifacts"]["manifest"]
    report = dict(
        graphs={},
        limitations=(
            "Static dependency and compiler declaration comparison. Neither runtime precision, "
            "collective execution order nor numerical causality is established. Continuation has external first A."
        ),
    )
    for group in ("q_proj", "k_proj", "v_proj"):
        for kind in ("identical_inner", "products", "continuation"):
            for layout in ("replicated", "direction_sharded"):
                key = f"{group}/{kind}/{layout}"
                texts = {}
                for ir, artifact in manifest[key].items():
                    data = StoragePath(artifact["uri"]).read_bytes(requests_timeout=60)
                    assert len(data) == artifact["compressed_bytes"]
                    assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
                    content = gzip.decompress(data)
                    local = args.local_dir / (key.replace("/", "__") + f".{ir}.txt")
                    assert content == local.read_bytes(), local
                    texts[ir] = content.decode()
                fields = re.findall(r'jax.result_info = "result\[\x27([^\x27]+)\x27\]"', texts["stablehlo"])
                if kind == "products":
                    assert fields == ["A", "B", "C", "D", "X1", "final", "norm", "normalized_input"]
                    path = (fields.index("final"),)
                else:
                    assert not fields
                    path = ()
                graph = aligned_products(texts["optimized_hlo"], path, kind)
                graph.update(artifacts=manifest[key], sha256_size_and_local_bytes_verified=True, output_path=list(path))
                report["graphs"][key] = graph
                print(key, len(graph["products"]), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.summary.write_text(json.dumps(compact_report(report), indent=2) + "\n")


if __name__ == "__main__":
    main()
