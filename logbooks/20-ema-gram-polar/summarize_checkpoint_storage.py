# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count logical checkpoint array storage from already-cached manifest JSON."""

import argparse
import hashlib
import json
import math
from pathlib import Path

DTYPE_BYTES = {"float32": 4, "int32": 4, "int64": 8, "uint32": 4, "bfloat16": 2}


def summarize(path):
    raw = path.read_bytes()
    document = json.loads(raw)
    manifest = document.get("parsed", document)
    counts = {}
    for array in manifest["arrays"]:
        name = array["path"]
        category = "other"
        if name.startswith("model/"):
            category = "model"
        elif "/matrices/" in name:
            category = "matrix_" + name.rsplit("/", 1)[-1]
        elif "/adamw/" in name:
            category = "adamw"
        size = math.prod(array["shape"]) * DTYPE_BYTES[array["dtype"]]
        counts[category] = counts.get(category, 0) + size
    return dict(manifest_sha256=hashlib.sha256(raw).hexdigest(), logical_bytes=counts, total_bytes=sum(counts.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--quarter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = dict(
        scope=(
            "Logical checkpoint array bytes from cached manifests; "
            "not allocator peak, compressed object bytes, or per-device memory"
        ),
        sources={"alpha0": summarize(args.baseline), "alpha025": summarize(args.quarter)},
        remote_reads=0,
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
