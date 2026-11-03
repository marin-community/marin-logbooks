# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare saved normalization-control compiler paths without device execution."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from inspect_continuation_paths import aligned_products, backend_config
from inspect_product_paths import output_dependency_graph
from rigging.filesystem.storage_path import StoragePath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--continuation-completion", type=Path, required=True)
    parser.add_argument("--continuation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    completion = json.loads(args.completion.read_text())
    previous_completion = json.loads(args.continuation_completion.read_text())
    raw = args.report.read_bytes()
    previous_raw = args.continuation_report.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == completion["report_sha256"]
    assert hashlib.sha256(previous_raw).hexdigest() == previous_completion["report_sha256"]
    numerical, previous = json.loads(raw), json.loads(previous_raw)
    manifest = completion["compiler_artifacts"]["manifest"]
    result = dict(graphs={}, comparisons={}, cross_job_fingerprints={}, backend_configs={}, consumed_files=0)

    def config_ref(row):
        config = backend_config(row)
        if config is None:
            return None
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        result["backend_configs"][digest] = config
        return digest

    for group in ("q_proj", "k_proj", "v_proj"):
        for kind in ("identical_inner", "normalized_polar", "raw_first_A", "normalized_first_A"):
            for layout in ("replicated", "direction_sharded"):
                key = (
                    f"{group}/{kind}/{layout}" if kind == "identical_inner" else f"{group}/normalization/{kind}/{layout}"
                )
                texts = {}
                for ir, artifact in manifest[key].items():
                    data = StoragePath(artifact["uri"]).read_bytes(requests_timeout=60)
                    assert len(data) == artifact["compressed_bytes"]
                    assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
                    decoded = gzip.decompress(data)
                    assert decoded == (args.local_dir / (key.replace("/", "__") + f".{ir}.txt")).read_bytes()
                    texts[ir] = decoded.decode()
                    result["consumed_files"] += 1
                graph = output_dependency_graph(texts["optimized_hlo"], ())
                rows = graph["records"]
                products = []
                if kind in ("identical_inner", "normalized_polar"):
                    aligned = aligned_products(texts["optimized_hlo"], (), kind)
                    for product in aligned["products"]:
                        row = product["operation"]
                        products.append(
                            dict(
                                iteration=product["iteration"],
                                product=product["product"],
                                operation=row["computation"] + "/" + row["name"],
                                operand_shapes=[r["shape"] for r in product["operands"]],
                                attributes_before_metadata=row["attributes"].split(", metadata=", 1)[0],
                                operation_config=config_ref(row),
                                calling_fusion_configs=[config_ref(r) for r in product["calling_fusions"]],
                                frontier_collectives=[
                                    dict(shape=r["shape"], instruction=r["instruction"])
                                    for r in product["operand_frontier"]
                                    if r["opcode"] in ("all-reduce", "all-gather", "reduce-scatter")
                                ],
                            )
                        )
                else:
                    operations = [r for r in rows if r["opcode"] in ("dot", "convolution")]
                    assert len(operations) == 1, key
                    row = operations[0]
                    expected_source = "normalization_first_gram" if kind == "raw_first_A" else "normalized_first_gram"
                    assert f"jit({expected_source})/vmap()/dot_general" in row["attributes"]
                    operands = {
                        r["name"]: r
                        for r in rows
                        if r["computation"] == row["computation"] and r["context"] == row["context"]
                    }
                    products.append(
                        dict(
                            operation=row["instruction"],
                            operand_shapes=[operands[name]["shape"] for name in row["operands"]],
                        )
                    )
                result["graphs"][key] = dict(
                    artifacts=manifest[key],
                    sha256_size_local_bytes_verified=True,
                    visible_products=products,
                    collectives=[
                        dict(shape=r["shape"], instruction=r["instruction"])
                        for r in rows
                        if r["opcode"] in ("all-reduce", "all-gather", "reduce-scatter")
                    ],
                    opaque_custom_calls=[r for r in rows if r["opcode"] == "custom-call"],
                    normalization_producers=[
                        r for r in rows if "jit(norm)" in r["attributes"] or "/vmap()/div" in r["attributes"]
                    ],
                )
        for layout in ("replicated", "direction_sharded"):
            left = result["graphs"][f"{group}/identical_inner/{layout}"]["visible_products"]
            right = result["graphs"][f"{group}/normalization/normalized_polar/{layout}"]["visible_products"]
            assert [(p["iteration"], p["product"]) for p in left] == [(p["iteration"], p["product"]) for p in right]
            result["comparisons"][f"{group}/{layout}"] = {
                field: [
                    f'{a["iteration"]}/{a["product"]}' for a, b in zip(left, right, strict=True) if a[field] != b[field]
                ]
                for field in (
                    "operand_shapes",
                    "attributes_before_metadata",
                    "operation_config",
                    "calling_fusion_configs",
                )
            }
        raw_equal = (
            numerical["stages"][f"{group}/inner"]["output_sha256"]
            == previous["stages"][f"{group}/inner"]["output_sha256"]
        )
        normalized_equal = (
            numerical["normalization_boundary_checks"][group]["normalized_input_sha256"]
            == previous["continuation_banks"][group]["replicated_A"]["x_sha256"]
        )
        assert raw_equal and normalized_equal, "Cross-job output comparisons require matching input fingerprints"
        result["cross_job_fingerprints"][group] = dict(
            raw_inner_equal=raw_equal,
            normalized_input_equal=normalized_equal,
            outputs_match_previous_replicated_A_continuation={
                layout: (
                    numerical["stages"][f"{group}/normalization/normalized_polar/{layout}/input"]["output_sha256"]
                    == previous["stages"][f"{group}/continuation/{layout}/replicated_A"]["output_sha256"]
                )
                for layout in ("replicated", "direction_sharded")
            },
        )
    result.update(
        report_sha256=completion["report_sha256"],
        previous_report_sha256=previous_completion["report_sha256"],
        limitations=(
            "Only visible HLO products are mapped. ConcatBitcast custom calls remain opaque; dependency traversal "
            "conservatively includes all declared operands. Static fields do not prove runtime precision, "
            "collective execution order, hidden tensor equality or numerical cause. Cross-job results are "
            "recorded fingerprint comparisons after verifying report bytes and matching input fingerprints."
        ),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                graphs=len(result["graphs"]),
                files=result["consumed_files"],
                comparisons=result["comparisons"],
                cross_job=result["cross_job_fingerprints"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
