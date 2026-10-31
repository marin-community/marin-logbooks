# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay the captured RMSNorm code with each observed launch configuration."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
import triton
from compare_moe_snapshots import compare_tensor
from vllm import ir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1
    manifest = json.loads(args.manifest.read_text())
    source = Path(manifest["source_file"]).read_bytes()
    assert all(
        hashlib.sha256(source).hexdigest() == record["source"]["sha256"] for record in manifest["experiments"].values()
    )
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {"reference": args.reference, "candidate": args.candidate}
    records = {
        label: [
            torch.load(root / f"latent-projection-{index}.pt", weights_only=True, map_location="cpu")
            for index in range(2)
        ]
        for label, root in inputs.items()
    }
    tolerance = ir.ops.rms_norm.get_tolerance(torch.bfloat16)
    rows = []
    for label, record in manifest["experiments"].items():
        module_path = args.output / f"captured_kernel_{label}.py"
        module_path.write_bytes(source)
        spec = importlib.util.spec_from_file_location(f"captured_kernel_{label}", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        kernel = getattr(module, record["kernel"]["name"])
        (selected,) = record["kernel"]["launchers"]
        kernel.configs = [
            triton.Config(selected["kwargs"], num_warps=selected["num_warps"], num_stages=selected["num_stages"])
        ]
        kernel.inductor_meta["dynamic_scale_rblock"] = False
        kernel.precompile()
        (launcher,) = kernel.launchers
        assert launcher.config.kwargs == selected["kwargs"]
        assert launcher.config.num_warps == selected["num_warps"]
        for index in range(2):
            reference, candidate = (records[key][index] for key in inputs)
            assert torch.equal(reference["projected"], candidate["projected"])
            assert torch.equal(reference["norm_weight"], candidate["norm_weight"])
            projected = reference["projected"].cuda()
            weight = reference["norm_weight"].cuda()
            kernel.run(projected, weight, *projected.shape, stream=torch.cuda.current_stream().cuda_stream)
            actual = projected.cpu()
            expected = torch.load(
                inputs[label] / f"moe-snapshot-dp0-model.layers.0.mlp.experts-{index}.pt",
                weights_only=True,
                map_location="cpu",
            )["hidden"]
            native_reference = ir.ops.rms_norm.impls["native"].impl_fn(
                reference["projected"], reference["norm_weight"], reference["variance_epsilon"]
            )
            torch.testing.assert_close(actual, native_reference, **tolerance)
            comparison = compare_tensor(expected, actual)
            assert comparison["exact"], comparison
            rows.append(
                {
                    "configuration": label,
                    "prefill_index": index,
                    "replay_vs_matching_capture": comparison,
                    "native_reference_tolerance": tolerance,
                    "native_reference_passed": True,
                    "replay_cache_hash": launcher.cache_hash,
                    "captured_cache_hash": selected["cache_hash"],
                }
            )
    result = {
        "status": "both_launch_configurations_reproduce_captured_normalization_exactly",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "kernel_source_sha256": hashlib.sha256(source).hexdigest(),
        "rows": rows,
        "scope": (
            "Same saved raw projection and weights for both configurations, two full2048x3072 prefills. "
            "Original captured kernel and native compilation path; launcher choices fixed to observed settings. "
            "Existing RMSNorm tolerance is unchanged. Numerical diagnostic only, no model or speed run."
        ),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
