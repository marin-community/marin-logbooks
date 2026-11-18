# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the installed canonicalization wheel through native FP8 MoE calls."""

import ast
import hashlib
import importlib
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import marlin_history_bootstrap as control
import torch
import vllm._custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    _canonicalize_marlin_moe_token_order,
    fused_marlin_moe,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_moe_layer_for_marlin
from vllm.scalar_type import scalar_types


def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    output = Path(payload["output"])
    module = importlib.import_module("vllm.model_executor.layers.fused_moe.experts.marlin_moe")
    model = importlib.import_module("vllm.model_executor.models.grugmoe")
    assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == payload["marlin_sha256"]
    assert hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest() == payload["model_sha256"]
    assert hashlib.sha256(Path(control.__file__).read_bytes()).hexdigest() == payload["control_sha256"]
    assert importlib.metadata.version("vllm") == payload["candidate_manifest"]["version"]
    assert torch.__version__ == "2.14.1+cu132"
    record = {
        "scope": (
            "Actual cold-installed candidate wheel; native FP8 components and raw CUDA graphs, no serving/KV qualification"
        ),
        "status": "running",
        "source_commit": payload["source_commit"],
        "research_source": payload["research_source"],
        "wheel_sha256": payload["candidate_manifest"]["wheels"][0]["sha256"],
        "marlin_module": module.__file__,
        "marlin_sha256": payload["marlin_sha256"],
        "model_sha256": payload["model_sha256"],
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "started_unix": time.time(),
        "cases": [],
        "graphs": [],
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
    tests = ast.parse((Path(payload["source_root"]) / "tests/kernels/moe/test_moe.py").read_text())
    regression = next(
        node
        for node in tests.body
        if isinstance(node, ast.FunctionDef) and node.name == "test_canonicalize_marlin_moe_token_order"
    )
    namespace = {"torch": torch, "_canonicalize_marlin_moe_token_order": _canonicalize_marlin_moe_token_order}
    exec(compile(ast.Module(body=[regression], type_ignores=[]), "native-canonical-regression", "exec"), namespace)
    namespace[regression.name]()
    record["hostile_inactive_tail_regression"] = "passed"
    output.write_text(json.dumps(record, indent=2) + "\n")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    local, global_experts, hidden, intermediate, topk, maximum = 24, 384, 3072, 3072, 8, 2048
    with torch.inference_mode(), set_current_vllm_config(VllmConfig()):
        inputs = (torch.randn((maximum, hidden), device="cuda") / 2048).bfloat16()
        magnitude = 2.0 ** (torch.arange(hidden, device="cuda") % 9 - 4)
        w13 = (
            torch.randn((local, 2 * intermediate, hidden), device="cuda", dtype=torch.bfloat16)
            * magnitude
            / math.sqrt(hidden)
        ).bfloat16()
        w2 = (
            torch.randn((local, hidden, intermediate), device="cuda", dtype=torch.bfloat16)
            * magnitude
            / math.sqrt(intermediate)
        ).bfloat16()
        q13, s13 = control.quantize(w13, exact=False)
        q2, s2 = control.quantize(w2, exact=False)
        layer = SimpleNamespace(
            num_experts=local,
            hidden_size=hidden,
            intermediate_size_per_partition=intermediate,
            orig_dtype=torch.bfloat16,
            w13_weight=w13,
            weight_block_size=None,
        )
        q13, q2, s13, s2 = prepare_fp8_moe_layer_for_marlin(layer, q13, q2, s13, s2)
        probabilities = torch.tensor([1, 2, 3, 4, 4, 3, 2, 1], device="cuda", dtype=torch.float32) / 20
        probabilities = probabilities.expand(maximum, -1).contiguous()
        expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device="cuda")
        expert_map[:local] = torch.arange(local, dtype=torch.int32, device="cuda")
        cache13 = torch.empty(maximum * topk * 2 * intermediate, dtype=torch.bfloat16, device="cuda")
        cache2 = torch.empty(maximum * topk * intermediate, dtype=torch.bfloat16, device="cuda")
        out = torch.empty((maximum, hidden), dtype=torch.bfloat16, device="cuda")

        def run(x, ids, weights):
            return fused_marlin_moe(
                x,
                q13,
                q2,
                None,
                None,
                s13,
                s2,
                weights,
                ids,
                scalar_types.float8_e4m3fn.id,
                global_num_experts=global_experts,
                expert_map=expert_map,
                workspace=layer.workspace,
                intermediate_cache13=cache13,
                intermediate_cache2=cache2,
                output=out[: len(x)],
            ).clone()

        for count in (1, 16, 115, 128, 2048):
            for route in ("hot-local-eight", "hot-local-four-mixed"):
                ids = torch.arange(topk, device="cuda").expand(count, -1).contiguous()
                if route == "hot-local-four-mixed":
                    ids[:, 4:] += local
                x = (inputs[:count] * magnitude).bfloat16()
                weights = probabilities[:count]
                estimated = math.ceil(count * local / global_experts)
                for block in (8, 16, 32, 48, 64):
                    if estimated * topk / local / block < 0.9:
                        break
                original, experts, padded = moe_align_block_size(
                    ids, block, global_experts, expert_map, ignore_invalid_experts=True
                )
                canonical = _canonicalize_marlin_moe_token_order(original, experts, padded, block, ids.numel())
                _, contributions = control.layout_outputs(
                    x, ids, weights, expert_map, q13, q2, s13, s2, layer, canonical, experts, padded, block
                )
                expected = torch.zeros_like(x)
                ops.moe_sum(contributions.view(count, topk, hidden), expected, ids, expert_map)
                actual = run(x, ids, weights)
                repeated = run(x, ids, weights)
                row = {
                    "tokens": count,
                    "routing": route,
                    "canonical_reference_max": control.maximum_difference(actual, expected),
                    "repeat_max": control.maximum_difference(actual, repeated),
                    "nonzero_output": bool(torch.count_nonzero(actual).item()),
                }
                row["passed"] = row["canonical_reference_max"] == row["repeat_max"] == 0 and row["nonzero_output"]
                record["cases"].append(row)
                output.write_text(json.dumps(record, indent=2) + "\n")
                assert row["passed"], row
                if count in (128, 2048) and route == "hot-local-four-mixed":
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = run(x, ids, weights)
                    graph.replay()
                    torch.cuda.synchronize()
                    graph_row = {"tokens": count, "eager_max": control.maximum_difference(actual, captured)}
                    graph_row["passed"] = graph_row["eager_max"] == 0
                    record["graphs"].append(graph_row)
                    output.write_text(json.dumps(record, indent=2) + "\n")
                    assert graph_row["passed"], graph_row
    record |= {
        "status": "passed",
        "finished_unix": time.time(),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "cases": len(record["cases"]), "graphs": len(record["graphs"])}))


if __name__ == "__main__":
    main()
