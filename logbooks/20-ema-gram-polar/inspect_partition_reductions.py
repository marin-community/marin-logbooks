# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare saved first-product producers and reduction configurations."""

import argparse
import gzip
import hashlib
import json
import re
from functools import cache
from pathlib import Path

from compiler_operation_inventory import COMPUTATION, INSTRUCTION, OPERAND
from inspect_continuation_paths import aligned_products, backend_config
from inspect_product_paths import output_dependency_graph
from rigging.filesystem.storage_path import StoragePath


def producer_signatures(text):
    """Hash rooted instruction DAGs, keeping argument order and called computations."""
    scopes, roots = {}, {}
    for line in text.splitlines():
        header = COMPUTATION.match(line)
        if header:
            scope = header[1]
            scopes[scope] = {}
        match = INSTRUCTION.match(line)
        if match:
            name, shape, opcode, operands, attributes = match.groups()
            scopes[scope][name] = (shape, opcode, operands, attributes)
            if line.lstrip().startswith("ROOT "):
                roots[scope] = name

    @cache
    def signature(scope, name):
        shape, opcode, operands, attributes = scopes[scope][name]
        ordered = OPERAND.findall(operands)
        # Preserve literals (constants and parameter indices), argument order and duplicates.
        operand_form = re.sub(r"%[\w.-]+", "@operand", operands)
        attrs = re.sub(r", metadata=\{[^}]*\}", "", attributes)
        called = []
        for target in re.findall(r"(?:calls|to_apply)=%([\w.-]+)", attrs):
            called.append(signature(target, roots[target]))
        attrs = re.sub(r"((?:calls|to_apply)=)%[\w.-]+", r"\1@computation", attrs)
        payload = [shape, opcode, operand_form, attrs, [signature(scope, child) for child in ordered], called]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    return signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    completion = json.loads(args.completion.read_text())
    manifest = completion["compiler_artifacts"]["manifest"]
    result = dict(graphs={}, comparisons={}, consumed_files=0)
    for group in ("q_proj", "k_proj", "v_proj"):
        for kind in ("raw_first_A", "normalized_first_A", "identical_inner"):
            for layout in ("replicated", "direction_sharded"):
                key = (
                    f"{group}/identical_inner/{layout}"
                    if kind == "identical_inner"
                    else f"{group}/partition/{kind}/{layout}"
                )
                texts = {}
                for ir, artifact in manifest[key].items():
                    raw = StoragePath(artifact["uri"]).read_bytes(requests_timeout=60)
                    assert len(raw) == artifact["compressed_bytes"]
                    assert hashlib.sha256(raw).hexdigest() == artifact["sha256"]
                    decoded = gzip.decompress(raw)
                    assert decoded == (args.local_dir / (key.replace("/", "__") + f".{ir}.txt")).read_bytes()
                    texts[ir] = decoded.decode()
                    result["consumed_files"] += 1
                graph = output_dependency_graph(texts["optimized_hlo"], ())
                rows = {row["identity"]: row for row in graph["records"]}
                edges = graph["dependencies"]
                if kind == "identical_inner":
                    aligned = aligned_products(texts["optimized_hlo"], (), kind)
                    product = aligned["products"][0]["operation"]
                    assert aligned["products"][0]["iteration"] == 1 and aligned["products"][0]["product"] == "A"
                else:
                    products = [row for row in rows.values() if row["opcode"] in ("dot", "convolution")]
                    assert len(products) == 1
                    product = products[0]
                    expected = "raw_contracting_gram" if kind == "raw_first_A" else "contracting_product"
                    assert f"jit({expected})/dot_general" in product["attributes"]
                assert "dim_labels=0bf_0oi->0bf" in product["attributes"]
                parents = []
                for ctx in product["context"]:
                    scope, name = ctx.split("/", 1)
                    matches = [row for row in rows.values() if row["computation"] == scope and row["name"] == name]
                    assert len(matches) == 1
                    parents.append(matches[0])
                signature = producer_signatures(texts["optimized_hlo"])
                outer = parents[0] if parents else product
                # Find nearest reductions consuming this product without crossing another product.
                reverse = {node: set() for node in rows}
                for parent, children in edges.items():
                    for child in children:
                        reverse[child].add(parent)
                frontier, seen, reductions = [product["identity"]], set(), []
                while frontier:
                    node = frontier.pop()
                    if node in seen:
                        continue
                    seen.add(node)
                    row = rows[node]
                    if node != product["identity"] and row["opcode"] in ("dot", "convolution"):
                        continue
                    if row["opcode"] in ("all-reduce", "reduce-scatter"):
                        reductions.append(row)
                        continue
                    frontier.extend(reverse[node])
                reduction_records = []
                for row in reductions:
                    callers = []
                    for ctx in row["context"]:
                        scope, name = ctx.split("/", 1)
                        matches = [r for r in rows.values() if r["computation"] == scope and r["name"] == name]
                        assert len(matches) == 1
                        callers.append(matches[0])
                    configs = [(row, backend_config(row)), *((p, backend_config(p)) for p in callers)]
                    reduction_records.append(
                        dict(
                            instruction=row["instruction"],
                            calling_fusions=callers,
                            collective_configs=[
                                dict(
                                    location=p["computation"] + "/" + p["name"], config=c["collective_algorithm_config"]
                                )
                                for p, c in configs
                                if c and "collective_algorithm_config" in c
                            ],
                        )
                    )
                same_scope = {
                    r["name"]: r
                    for r in rows.values()
                    if r["context"] == product["context"] and r["computation"] == product["computation"]
                }
                result["graphs"][key] = dict(
                    artifacts=manifest[key],
                    sha_size_localbytes_verified=True,
                    product=product,
                    operand_declarations=[same_scope[n]["shape"] for n in product["operands"]],
                    calling_fusions=parents,
                    rooted_outer_producer_sha256=signature(outer["computation"], outer["name"]),
                    nearest_product_reductions=reduction_records,
                    normalization_records=[
                        r for r in rows.values() if "jit(norm)" in r["attributes"] or "/vmap()/div" in r["attributes"]
                    ],
                )
        for kind in ("raw_first_A", "normalized_first_A"):
            a = result["graphs"][f"{group}/partition/{kind}/replicated"]
            b = result["graphs"][f"{group}/partition/{kind}/direction_sharded"]
            result["comparisons"][f"{group}/{kind}/layouts"] = dict(
                rooted_outer_producer_equal=a["rooted_outer_producer_sha256"] == b["rooted_outer_producer_sha256"],
                operand_declarations_equal=a["operand_declarations"] == b["operand_declarations"],
                product_parent_configs_equal=[backend_config(r) for r in a["calling_fusions"]]
                == [backend_config(r) for r in b["calling_fusions"]],
                reduction_algorithms={
                    layout: [x["config"] for red in obj["nearest_product_reductions"] for x in red["collective_configs"]]
                    for layout, obj in [("replicated", a), ("direction_sharded", b)]
                },
            )
    result["reference_comparisons"] = {}
    for group in ("q_proj", "k_proj", "v_proj"):
        reference = result["graphs"][f"{group}/identical_inner/direction_sharded"]
        for kind in ("raw_first_A", "normalized_first_A"):
            for layout in ("replicated", "direction_sharded"):
                key = f"{group}/partition/{kind}/{layout}"
                value = result["graphs"][key]
                result["reference_comparisons"][key] = dict(
                    reference=f"{group}/identical_inner/direction_sharded/first_A",
                    rooted_outer_producer_equal=value["rooted_outer_producer_sha256"]
                    == reference["rooted_outer_producer_sha256"],
                    product_parent_configs_equal=[backend_config(r) for r in value["calling_fusions"]]
                    == [backend_config(r) for r in reference["calling_fusions"]],
                )
    result["signature_scope"] = (
        "Rooted product-producing DAG, operands and called computations; includes constants, parameter indices, "
        "shapes, ordered repeated arguments and attributes except metadata. SSA/computation names normalized. "
        "Not runtime value or schedule equivalence."
    )
    result["limitations"] = (
        "Static declarations and configurations do not establish runtime precision, collective order, hidden "
        "value equality or numerical cause. Reference products are mapped by dependency chain, not text order."
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(graphs=len(result["graphs"]), files=result["consumed_files"], comparisons=result["comparisons"]),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
