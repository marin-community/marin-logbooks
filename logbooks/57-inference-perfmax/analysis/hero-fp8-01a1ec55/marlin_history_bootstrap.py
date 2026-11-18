# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test native Marlin EP filtering and repeated outputs after different work."""

import hashlib
import importlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import vllm._custom_ops as ops
from vllm.config import ensure_current_vllm_config
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.online.fp8 import _fp8_scale
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_moe_layer_for_marlin
from vllm.scalar_type import scalar_types


def quantize(weight):
    scales = _fp8_scale(weight.float().abs().flatten(1).amax(dim=1))
    packed = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    for expert in range(weight.shape[0]):
        packed[expert] = ops.scaled_fp8_quant(weight[expert], scale=scales[expert])[0]
    assert torch.equal(packed.to(weight.dtype) * scales[:, None, None], weight)
    return packed, scales


def reference(x, ids, probabilities, expert_map, w13, w2):
    local_ids = expert_map[ids.clamp_min(0)]
    local_ids = torch.where(ids >= 0, local_ids, -1)
    result = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(w13.shape[0]):
        rows, slots = torch.where(local_ids == expert)
        if not rows.numel():
            continue
        gate, up = (x[rows].float() @ w13[expert].float().T).bfloat16().chunk(2, dim=-1)
        activation = (F.silu(gate.float()) * up.float()).bfloat16()
        down = activation.float() @ w2[expert].float().T
        weighted = (down * probabilities[rows, slots, None]).bfloat16().float()
        result.index_add_(0, rows, weighted)
    return result.bfloat16()


def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    output = Path(payload["output"])
    module = importlib.import_module("vllm.model_executor.layers.fused_moe.experts.marlin_moe")
    native_hash = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    assert native_hash == payload["marlin_sha256"]
    assert importlib.metadata.version("vllm") == payload["candidate_manifest"]["version"]
    assert torch.__version__ == "2.14.1+cu132"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    record = {
        "status": "running",
        "scope": "One-GPU component diagnostic: native Marlin local EP contributions; no serving/KV qualification",
        "source_commit": payload["source_commit"],
        "wheel_sha256": payload["candidate_manifest"]["wheels"][0]["sha256"],
        "marlin_module": module.__file__,
        "marlin_sha256": native_hash,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "started_unix": time.time(),
        "cases": [],
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
    local, global_experts, hidden, intermediate, topk, maximum = 24, 384, 3072, 3072, 8, 2048
    with torch.inference_mode(), ensure_current_vllm_config():
        levels = torch.tensor([0.21875, 0.4375, 0.875, 1.75], device="cuda")
        expert = torch.arange(local, device="cuda")[:, None, None]
        rows = torch.arange(2 * intermediate, device="cuda")[None, :, None]
        columns = torch.arange(hidden, device="cuda")[None, None, :]
        w13 = (
            levels[(rows // 8 + 2 * (columns // 16) + expert) % 4]
            * torch.where((rows // 16 + columns // 32 + expert) % 2 == 0, 1, -1)
            * (2.0 ** (expert % 3 - 1))
        ).bfloat16()
        rows = torch.arange(hidden, device="cuda")[None, :, None]
        columns = torch.arange(intermediate, device="cuda")[None, None, :]
        w2 = (
            levels[(2 * (rows // 16) + columns // 8 + expert) % 4]
            * torch.where((rows // 32 + columns // 16 + expert) % 2 == 0, 1, -1)
            * (2.0 ** (1 - expert % 3))
        ).bfloat16()
        q13, s13 = quantize(w13)
        q2, s2 = quantize(w2)
        layer = SimpleNamespace(
            num_experts=local,
            hidden_size=hidden,
            intermediate_size_per_partition=intermediate,
            orig_dtype=torch.bfloat16,
            w13_weight=w13,
            weight_block_size=None,
        )
        q13, q2, s13, s2 = prepare_fp8_moe_layer_for_marlin(layer, q13, q2, s13, s2)
        cache13 = torch.empty(maximum * topk * 2 * intermediate, dtype=torch.bfloat16, device="cuda")
        cache2 = torch.empty(maximum * topk * intermediate, dtype=torch.bfloat16, device="cuda")
        out = torch.empty((maximum, hidden), dtype=torch.bfloat16, device="cuda")
        inputs = (torch.randn((maximum, hidden), device="cuda") / 2048).bfloat16()
        probabilities = torch.tensor([1, 2, 3, 4, 4, 3, 2, 1], device="cuda", dtype=torch.float32) / 20
        probabilities = probabilities.expand(maximum, -1).contiguous()
        all_rows = torch.arange(maximum, device="cuda")[:, None]
        slots = torch.arange(topk, device="cuda")[None, :]
        for offset in (0, global_experts - local):
            expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device="cuda")
            expert_map[offset : offset + local] = torch.arange(local, dtype=torch.int32, device="cuda")
            outside = local if offset == 0 else 0
            routes = {
                "all-local": (all_rows + slots) % local + offset,
                "mixed": torch.where(slots < 4, (all_rows + slots) % local + offset, (all_rows + slots) % 24 + outside),
                "no-local": (all_rows + slots) % 24 + outside,
                "negative-padding": torch.where(all_rows % 3 == 0, -1, (all_rows + slots) % local + offset),
            }

            def run(x, ids, weights, expert_map=expert_map):
                return fused_marlin_moe(
                    x,
                    q13,
                    q2,
                    None,
                    None,
                    s13,
                    s2,
                    weights,
                    ids.contiguous(),
                    scalar_types.float8_e4m3fn.id,
                    global_num_experts=global_experts,
                    expert_map=expert_map,
                    workspace=layer.workspace,
                    intermediate_cache13=cache13,
                    intermediate_cache2=cache2,
                    output=out[: len(x)],
                ).clone()

            for name, ids in routes.items():
                for count in (1, 7, 8, 127, 128, 129, 512, 2048):
                    x, target, weights = inputs[:count], ids[:count], probabilities[:count]
                    cache13.fill_(16)
                    cache2.fill_(-16)
                    out.fill_(32)
                    before = run(x, target, weights)
                    expected = reference(x, target, weights, expert_map, w13, w2)
                    error = (before.float() - expected.float()).abs()
                    row = {
                        "local_offset": offset,
                        "routing": name,
                        "tokens": count,
                        "reference_max_abs": error.max().item(),
                        "reference_rms": error.square().mean().sqrt().item(),
                        "output_max_abs": before.float().abs().max().item(),
                    }
                    record["cases"].append(row)
                    output.write_text(json.dumps(record, indent=2) + "\n")
                    torch.testing.assert_close(before.float(), expected.float(), atol=0.02, rtol=0.02)
                    run(inputs.flip(0).contiguous(), routes["mixed"].flip(0).contiguous(), probabilities)
                    after = run(x, target, weights)
                    row["history_max_abs"] = (before.float() - after.float()).abs().max().item()
                    assert torch.equal(before, after), row
                    if name == "no-local":
                        assert not torch.count_nonzero(before), row
                    if name == "negative-padding":
                        assert not torch.count_nonzero(before[torch.arange(count, device="cuda") % 3 == 0]), row
                    row["status"] = "passed"
                    output.write_text(json.dumps(record, indent=2) + "\n")
        assert any(x["output_max_abs"] > 0.01 for x in record["cases"])
    record |= {"status": "passed", "finished_unix": time.time(), "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "cases": len(record["cases"])}))


if __name__ == "__main__":
    main()
