# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure CPU reconstruction error on a few real expert down projections."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import ml_dtypes
import numpy as np
from rigging.filesystem.buckets import filesystem_for


def read_range(uri: str, start: int, end: int) -> bytes:
    if uri.startswith("https://"):
        request = urllib.request.Request(uri, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(request, timeout=30) as stream:
            assert stream.status == 206
            data = stream.read(end - start)
    else:
        fs, path = filesystem_for(uri)
        data = fs.cat_file(path, start=start, end=end)
    assert len(data) == end - start
    return data


def read_expert(base: str, index: dict, name: str, expert: int | None) -> tuple[np.ndarray, dict]:
    uri = f"{base}/{index[name]}"
    length = int.from_bytes(read_range(uri, 0, 8), "little")
    assert 0 < length < 4 * 1024**2
    info = json.loads(read_range(uri, 8, 8 + length))[name]
    start, end = info["data_offsets"]
    shape = info["shape"]
    dtype = {"F32": "<f4", "BF16": "<u2"}[info["dtype"]]
    if expert is not None:
        stride = (end - start) // shape[0]
        start += expert * stride
        end = start + stride
        shape = shape[1:]
    raw = read_range(uri, 8 + length + start, 8 + length + end)
    weight = np.frombuffer(raw, dtype=dtype).reshape(shape)
    if info["dtype"] == "BF16":
        weight = (weight.astype(np.uint32) << 16).view(np.float32)
    else:
        weight = weight.astype(ml_dtypes.bfloat16).astype(np.float32)
    return weight, {"uri": uri, "tensor": name, "expert": expert, "sha256": hashlib.sha256(raw).hexdigest()}


def reconstruct(groups, axis):
    max_fp8 = 448.0
    scale = np.maximum(np.max(np.abs(groups), axis=axis, keepdims=True), 1e-4) / max_fp8
    normalized = np.clip(groups / scale, -max_fp8, max_fp8)
    quantized = normalized.astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
    # The installed Marlin packer casts scales to the original model dtype.
    return quantized * scale.astype(ml_dtypes.bfloat16).astype(np.float32)


def metrics(weight, candidate, inputs):
    diff = candidate - weight
    baseline = weight @ inputs
    predicted = candidate @ inputs
    return {
        "relative_weight_l2": float(np.linalg.norm(diff) / np.linalg.norm(weight)),
        "relative_matmul_l2": float(np.linalg.norm(predicted - baseline) / np.linalg.norm(baseline)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indexes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sampling = {
        "snowball": (
            "https://huggingface.co/open-athena/Marin-Snowball-67B-A2B-v0.1.1/resolve/59852d6f07261722bbec1edb6ab148317156b58e",
            [(0, 0), (13, 128), (25, 255)],
        ),
        "mochi": (
            "s3://marin-us-east-02a/marin/exports/mochi-sft/hero-sft-fullab-step393k-v2/step-400407/2af3f828aa-cpu-a1",
            [(0, 0), (24, 192), (47, 383)],
        ),
    }
    random = np.random.default_rng(42)
    results = {}
    for model, (base, samples) in sampling.items():
        index = json.loads((args.indexes / f"{model}-index.json").read_text())["weight_map"]
        results[model] = []
        for layer, expert in samples:
            prefix = f"model.layers.{layer}.mlp.experts"
            if model == "mochi":
                name = f"{prefix}.{expert}.down_proj.weight"
                weight, receipt = read_expert(base, index, name, None)
            else:
                name = f"{prefix}.down_proj.weight"
                weight, receipt = read_expert(base, index, name, expert)
            output_width, input_width = weight.shape
            tiles = weight.reshape(output_width // 128, 128, input_width // 128, 128)
            inputs = random.standard_normal((input_width, 8), dtype=np.float32)
            result = {
                "layer": layer,
                "expert": expert,
                "receipt": receipt,
                "per_tensor": metrics(weight, reconstruct(weight, (0, 1)), inputs),
                "per_block": metrics(weight, reconstruct(tiles, (1, 3)).reshape(weight.shape), inputs),
                "per_channel": metrics(weight, reconstruct(weight, (1,)), inputs),
            }
            results[model].append(result)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
