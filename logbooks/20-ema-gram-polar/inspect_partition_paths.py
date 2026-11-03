# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from inspect_product_paths import output_ancestors
from rigging.filesystem.storage_path import StoragePath


def main():
    parser = argparse.ArgumentParser(description="Verify first-product partitions from saved compiler artifacts")
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    completion = json.loads(args.completion.read_text())
    root = args.local_dir
    out = {
        "graphs": {},
        "limitation": "Static declared contraction and collective ancestry; no runtime precision or causal attribution.",
    }
    for name in ("q_proj", "k_proj", "v_proj"):
        for boundary in ("raw", "normalized"):
            for layout in ("replicated", "direction_sharded"):
                key = f"{name}/partition/{boundary}_first_A/{layout}"
                artifact = completion["compiler_artifacts"]["manifest"][key]["optimized_hlo"]
                data = StoragePath(artifact["uri"]).read_bytes(requests_timeout=60)
                assert (
                    len(data) == artifact["compressed_bytes"] and hashlib.sha256(data).hexdigest() == artifact["sha256"]
                )
                text = gzip.decompress(data).decode()
                assert text == (root / (key.replace("/", "__") + ".optimized_hlo.txt")).read_text()
                graph = output_ancestors(text, ())
                product = graph["matrix_product"]
                records = graph["records"]
                operands = [
                    next(
                        x
                        for x in records
                        if x["context"] == product["context"]
                        and x["computation"] == product["computation"]
                        and x["name"] == operand
                    )
                    for operand in product["operands"]
                ]
                shapes = [x["shape"] for x in operands]
                assert all(shape.startswith(("f32[11,1024,128]", "bf16[11,1024,128]")) for shape in shapes)
                assert "dim_labels=0bf_0oi->0bf" in product["attributes"]
                reductions = [x for x in graph["collectives"] if x["opcode"] in ("all-reduce", "reduce-scatter")]
                assert len(reductions) == 1 and reductions[0]["opcode"] == "all-reduce"
                out["graphs"][key] = dict(
                    artifact=artifact,
                    sha_size_localbytes_verified=True,
                    operand_declarations=shapes,
                    product=product,
                    collectives=graph["collectives"],
                    has_output_ancestry_reduction=bool(reductions),
                    source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                )
                print(key, shapes, [x["opcode"] for x in graph["collectives"]])
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
