# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record output-ancestor products and collectives from frozen-partial controls."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from inspect_continuation_paths import backend_config
from inspect_product_paths import output_dependency_graph
from rigging.filesystem.storage_path import StoragePath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    artifacts = {k: s["compiler_artifacts"] for k, s in report["stages"].items() if "compiler_artifacts" in s}
    artifacts.update({k: s["compiler_artifacts"] for k, s in report["fixed_executables"].items()})
    result = dict(graphs={}, consumed_files=0)
    for group in ("q_proj", "k_proj", "v_proj"):
        for suffix in ("local_partials", "reduce/replicated", "reduce/direction_sharded"):
            key = f"{group}/collectives/{suffix}"
            texts = {}
            for kind, artifact in artifacts[key].items():
                raw = StoragePath(artifact["uri"]).read_bytes(requests_timeout=60)
                assert len(raw) == artifact["compressed_bytes"]
                assert hashlib.sha256(raw).hexdigest() == artifact["sha256"]
                texts[kind] = gzip.decompress(raw).decode()
                result["consumed_files"] += 1
            graph = output_dependency_graph(texts["optimized_hlo"], ())
            rows = graph["records"]
            products = [r for r in rows if r["opcode"] in ("dot", "convolution")]
            collectives = [r for r in rows if r["opcode"] in ("all-reduce", "reduce-scatter")]
            if suffix == "local_partials":
                if report["config"]["smoke"] and not products:
                    # A one-column CPU contraction becomes an outer-product multiply.
                    assert report["config"]["shapes"][0][-1] == len(report["config"]["devices"])
                    products = [r for r in rows if r["opcode"] == "multiply" and "/dot_general" in r["attributes"]]
                    assert len(products) == 1
                    operands = products[0]["operands"]
                    assert len(operands) == 2 and len(set(operands)) == 2
                    assert all(any(r["name"] == op and r["opcode"] == "broadcast" for r in rows) for op in operands)
                assert len(products) == 1 and not collectives
            else:
                assert not products and len(collectives) == 1
            configs = [dict(identity=r["identity"], config=backend_config(r)) for r in rows if backend_config(r)]
            result["graphs"][key] = dict(
                artifacts=artifacts[key],
                products=products,
                collectives=collectives,
                backend_configs=configs,
                partial_bank=report["partial_banks"][group],
                fixed_executable=report["fixed_executables"][key],
            )
    result["limitations"] = (
        "Dependency-visible compiler records only. Collective names and configurations do not establish runtime "
        "precision, reduction order, original partial values or full optimizer equivalence."
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(graphs=len(result["graphs"]), files=result["consumed_files"])))


if __name__ == "__main__":
    main()
