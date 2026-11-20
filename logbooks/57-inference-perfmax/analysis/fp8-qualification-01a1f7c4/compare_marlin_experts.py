# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare Marlin with plain arithmetic on the same quantized expert weights."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.experts import marlin_moe
from vllm.model_executor.layers.quantization.online.fp8 import _fp8_scale
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_moe_layer_for_marlin
from vllm.scalar_type import scalar_types
from vllm.utils.deep_gemm import per_block_cast_to_fp8


def quantize_experts(weights, block_size):
    if block_size:
        quantized = [per_block_cast_to_fp8(weight, block_size) for weight in weights]
        values, scales = zip(*quantized, strict=True)
        return torch.stack(values), torch.stack(scales)
    scales = _fp8_scale(weights.flatten(1).abs().float().amax(dim=1))
    values = [ops.scaled_fp8_quant(weight, scale=scale)[0] for weight, scale in zip(weights, scales, strict=True)]
    return torch.stack(values), scales


def dequantize(values, scales, block_size):
    # The packer converts these same scales to BF16 before reordering.
    scales = scales.to(torch.bfloat16).float()
    if block_size:
        scales = scales.repeat_interleave(block_size[0], dim=1).repeat_interleave(block_size[1], dim=2)
    else:
        scales = scales.reshape(-1, 1, 1)
    return values.float() * scales


def reference(inputs, gate_up, down, weights, *, round_intermediates):
    """Use FP32 matmuls with optional rounding at the kernel's BF16 boundaries."""
    outputs = []
    for expert, (first, second) in enumerate(zip(gate_up, down, strict=True)):
        first_output = inputs.float() @ first.float().T
        if round_intermediates:
            first_output = first_output.bfloat16().float()
        gate, up = first_output.chunk(2, dim=1)
        intermediate = F.silu(gate) * up
        if round_intermediates:
            intermediate = intermediate.bfloat16().float()
        routed = (intermediate @ second.float().T) * weights[:, expert : expert + 1]
        outputs.append(routed.bfloat16() if round_intermediates else routed)
    return torch.stack(outputs, dim=1).sum(dim=1).float()


def difference(baseline, candidate):
    assert torch.isfinite(baseline).all() and torch.isfinite(candidate).all()
    residual = candidate.float() - baseline.float()
    return {
        "relative_l2": (torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(baseline)).item(),
        "max_abs": residual.abs().max().item(),
    }


@torch.inference_mode()
def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    assert hashlib.sha256(Path(marlin_moe.__file__).read_bytes()).hexdigest() == payload["marlin_sha256"]
    root = Path(payload["source_root"]) / "fp8_qualification_inputs"
    data_path = root / "weights.npz"
    with data_path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == payload["weights_sha256"]
    with np.load(data_path) as data:
        tensors = {key: torch.from_numpy(data[key]).view(torch.bfloat16).to("cuda") for key in data.files}
    gate_up = torch.cat((tensors["gate"], tensors["up"]), dim=1)
    down = tensors["down"]
    experts, width, intermediate = down.shape
    assert experts == 2 and width == intermediate == 3072
    torch.set_float32_matmul_precision("highest")
    generator = torch.Generator(device="cuda").manual_seed(42)
    samples = []
    for tokens in (1, 41, 257):
        inputs = torch.randn(tokens, width, generator=generator, device="cuda", dtype=torch.bfloat16)
        ids = torch.arange(experts, device="cuda", dtype=torch.int32).expand(tokens, -1).contiguous()
        weights = torch.randn(tokens, experts, generator=generator, device="cuda").sigmoid()
        weights *= 2.5 / weights.sum(dim=1, keepdim=True)
        baseline = reference(inputs, gate_up, down, weights, round_intermediates=True)
        samples.append((tokens, inputs, ids, weights, baseline))

    results = []
    for scheme, block_size in (("per_tensor", None), ("per_block", [128, 128])):
        first_values, first_scales = quantize_experts(gate_up, block_size)
        second_values, second_scales = quantize_experts(down, block_size)
        first = dequantize(first_values, first_scales, block_size)
        second = dequantize(second_values, second_scales, block_size)
        layer = SimpleNamespace(
            num_experts=experts,
            hidden_size=width,
            intermediate_size_per_partition=intermediate,
            orig_dtype=torch.bfloat16,
            w13_weight=first_values,
            weight_block_size=block_size,
        )
        w13, w2, s13, s2 = prepare_fp8_moe_layer_for_marlin(
            layer, first_values, second_values, first_scales, second_scales
        )
        for tokens, inputs, ids, weights, baseline in samples:
            oracle = reference(inputs, first, second, weights, round_intermediates=True)
            wide = reference(inputs, first, second, weights, round_intermediates=False)
            actual = marlin_moe.fused_marlin_moe(
                inputs,
                w13,
                w2,
                None,
                None,
                s13,
                s2,
                weights,
                ids,
                quant_type_id=scalar_types.float8_e4m3fn.id,
                workspace=layer.workspace,
            )
            assert actual.dtype == torch.bfloat16
            results.append(
                {
                    "scheme": scheme,
                    "rows": tokens,
                    "weight_cast_vs_bf16": difference(baseline, oracle),
                    "kernel_vs_same_quantized_values": difference(oracle, actual),
                    "intermediate_rounding": difference(wide, oracle),
                    "kernel_vs_bf16": difference(baseline, actual),
                }
            )
    output = {
        "source_commit": payload["source_commit"],
        "device": torch.cuda.get_device_name(),
        "scope": "Two fixed Mochi experts with synthetic BF16 inputs and fixed FP32 combine weights",
        "inputs": json.loads((root / "weights.json").read_text()),
        "results": results,
    }
    Path(payload["output"]).write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
