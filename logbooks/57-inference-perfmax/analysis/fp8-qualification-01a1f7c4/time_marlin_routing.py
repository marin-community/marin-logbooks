# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Time the installed Marlin route alignment with and without canonical ordering."""

import hashlib
import importlib
import json
import math
import statistics
import sys
from pathlib import Path

import torch
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size


def graph_micros(launch):
    for _ in range(5):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = launch()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(9):
        start, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            graph.replay()
        finish.record()
        finish.synchronize()
        samples.append(start.elapsed_time(finish) * 10)
    return statistics.median(samples), samples, result


def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    module = importlib.import_module("vllm.model_executor.layers.fused_moe.experts.marlin_moe")
    assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == payload["marlin_sha256"]
    canonicalize = module._canonicalize_marlin_moe_token_order
    hardware = payload["gpu_type"]
    experts, top_k, local = (256, 4, 32) if hardware == "H100" else (384, 8, 24)
    generator = torch.Generator(device="cuda").manual_seed(42)
    expert_map = torch.full((experts,), -1, device="cuda", dtype=torch.int32)
    expert_map[:local] = torch.arange(local, device="cuda", dtype=torch.int32)
    rows = []
    for tokens in (16, 128, 2048, 8192, 32768):
        scores = torch.randn((tokens, experts), device="cuda", generator=generator)
        topk_ids = scores.topk(top_k).indices.to(torch.int32)
        local_tokens = math.ceil(tokens * local / experts)
        for block_m in (8, 16, 32, 48, 64):
            if local_tokens * top_k / local / block_m < 0.9:
                break

        def align():
            return moe_align_block_size(topk_ids, block_m, experts, expert_map, ignore_invalid_experts=True)

        def align_and_order():
            slots, blocks, active = align()
            return canonicalize(slots, blocks, active, block_m, topk_ids.numel())

        baseline, before, layout = graph_micros(align)
        ordered, after, sorted_slots = graph_micros(align_and_order)
        slots, blocks, active = layout
        length = int(active.item())
        torch.testing.assert_close(
            slots[:length].sort().values,
            sorted_slots[:length].sort().values,
            rtol=0,
            atol=0,
        )
        rows.append({
            "input_rows_after_dispatch": tokens,
            "block_m": block_m,
            "buffer_slots": slots.numel(),
            "active_slots": length,
            "alignment_microseconds": baseline,
            "alignment_and_ordering_microseconds": ordered,
            "added_microseconds": ordered - baseline,
            "baseline_samples": before,
            "ordered_samples": after,
        })
    Path(payload["output"]).write_text(json.dumps({
        "scope": "Synthetic routes with the installed alignment and sorting code under CUDA graph replay",
        "source_commit": payload["source_commit"],
        "device": torch.cuda.get_device_name(),
        "experts": experts,
        "top_k": top_k,
        "local_experts": local,
        "results": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
