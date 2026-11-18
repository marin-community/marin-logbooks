# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test native Marlin EP filtering and repeated outputs after different work."""

import hashlib
import importlib
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import vllm._custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import _fused_marlin_moe, fused_marlin_moe
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.quantization.online.fp8 import _fp8_scale
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_moe_layer_for_marlin
from vllm.scalar_type import scalar_types


# Canonical layout control from vllm-project/vllm PR #52532 (Apache-2.0).
def _canonicalize_marlin_moe_token_order(
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    block_size_m: int,
    num_valid_tokens: int,
) -> torch.Tensor:
    """Canonicalize routed token IDs within contiguous expert regions.

    moe_align_block_size guarantees expert grouping but does not guarantee
    token ordering within an expert. Marlin's multi-threadblock split-K
    reduction can depend on physical block position, so equivalent routed
    layouts need a common ordering before entering the kernel.

    Alignment buffers may contain unspecified data after
    num_tokens_post_padded. Keep that inactive tail strictly after the active
    prefix so its contents cannot affect canonicalization.
    """
    block_experts = expert_ids.to(dtype=torch.int64)
    block_positions = torch.arange(
        expert_ids.numel(),
        device=expert_ids.device,
        dtype=torch.int64,
    )
    active_blocks = block_positions * block_size_m < num_tokens_post_padded

    region_starts = torch.cat(
        (
            torch.ones_like(block_experts[:1], dtype=torch.bool),
            (block_experts[1:] != block_experts[:-1]) | (active_blocks[1:] != active_blocks[:-1]),
        ),
        dim=0,
    )
    block_regions = torch.cumsum(region_starts.to(dtype=torch.int64), dim=0) - 1

    slot_regions = torch.repeat_interleave(
        block_regions,
        block_size_m,
    )[: sorted_token_ids.numel()]

    slot_positions = torch.arange(
        sorted_token_ids.numel(),
        device=sorted_token_ids.device,
        dtype=torch.int64,
    )
    active_slots = slot_positions < num_tokens_post_padded

    token_ids = sorted_token_ids.to(dtype=torch.int64)
    key_stride = num_valid_tokens + 1

    # Valid token IDs are in [0, num_valid_tokens), with the padding sentinel
    # at num_valid_tokens. Values in the inactive tail are unspecified, so do
    # not let them contribute to the sort key.
    token_key = torch.where(
        active_slots,
        token_ids,
        torch.zeros_like(token_ids),
    )
    sort_key = slot_regions * key_stride + token_key

    order = torch.argsort(sort_key)
    return sorted_token_ids.index_select(0, order)


def quantize(weight, exact=True):
    scales = _fp8_scale(weight.float().abs().flatten(1).amax(dim=1))
    packed = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    for expert in range(weight.shape[0]):
        packed[expert] = ops.scaled_fp8_quant(weight[expert], scale=scales[expert])[0]
    if exact:
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


def equivalent_layouts(sorted_ids, experts, post_padded, block, num_valid):
    used = int(post_padded.item())
    ids = sorted_ids[:used].cpu()
    groups = experts[: used // block].cpu().tolist()
    first, second = sorted_ids.clone(), sorted_ids.clone()
    begin = 0
    regions = 0
    while begin < len(groups):
        end = begin + 1
        while end < len(groups) and groups[end] == groups[begin]:
            end += 1
        lower, upper = begin * block, end * block
        tokens = ids[lower:upper]
        valid = tokens[tokens < num_valid].sort().values
        assert (valid >= 0).all()
        padding = torch.full((upper - lower - len(valid),), num_valid, dtype=torch.int32)
        first[lower:upper] = torch.cat((valid, padding)).to(sorted_ids.device)
        second[lower:upper] = torch.cat((valid.flip(0), padding)).to(sorted_ids.device)
        begin = end
        regions += 1
    return first, second, used, regions


def layout_outputs(x, ids, probabilities, expert_map, q13, q2, s13, s2, layer, order, experts, padded, block):
    count, hidden = x.shape
    topk = ids.shape[1]
    intermediate = layer.intermediate_size_per_partition
    c13 = torch.zeros(count * topk * 2 * intermediate, dtype=torch.bfloat16, device=x.device)
    c2 = torch.zeros(count * topk * intermediate, dtype=torch.bfloat16, device=x.device)
    output = torch.zeros((count * topk, hidden), dtype=torch.bfloat16, device=x.device)
    result = _fused_marlin_moe(
        hidden_states=x,
        w1=q13,
        w2=q2,
        bias1=None,
        bias2=None,
        w1_scale=s13,
        w2_scale=s2,
        topk_weights=probabilities,
        num_topk=topk,
        quant_type=scalar_types.float8_e4m3fn,
        apply_router_weight_on_input=False,
        expert_map=expert_map,
        block_size_m=block,
        sorted_token_ids=order,
        expert_ids=experts,
        num_tokens_post_padded=padded,
        topk_ids=ids,
        workspace=torch.zeros_like(layer.workspace),
        intermediate_cache13=c13,
        intermediate_cache2=c2,
        output=output,
    )
    torch.cuda.synchronize()
    return c13.clone(), result.clone()


def maximum_difference(first, second):
    return (first.float() - second.float()).abs().max().item()


def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    output = Path(payload["output"])
    module = importlib.import_module("vllm.model_executor.layers.fused_moe.experts.marlin_moe")
    native_hash = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    assert native_hash == payload["marlin_sha256"]
    model = importlib.import_module("vllm.model_executor.models.grugmoe")
    model_hash = hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest()
    assert model_hash == payload["model_sha256"]
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
        "model_module": model.__file__,
        "model_sha256": model_hash,
        "research_source": payload["research_source"],
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "multiprocessors": torch.cuda.get_device_properties(0).multi_processor_count,
        "capability": list(torch.cuda.get_device_capability()),
        "started_unix": time.time(),
        "cases": [],
        "cuda_graph_history": [],
        "completed_phases": [],
        "layout_order": [],
        "upstream_candidate": payload["upstream_candidate"],
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
    local, global_experts, hidden, intermediate, topk, maximum = 24, 384, 3072, 3072, 8, 2048
    with torch.inference_mode(), set_current_vllm_config(VllmConfig()):
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
        reused = payload.get("reuse_history")
        if reused:
            assert reused["source_commit"] == record["source_commit"]
            assert reused["wheel_sha256"] == record["wheel_sha256"]
            assert reused["marlin_sha256"] == record["marlin_sha256"]
            assert reused["torch"] == record["torch"]
            assert reused["device"] == record["device"]
            record["reused_history_evidence"] = reused
            record["completed_phases"].append("history-reused-from-a2")
            output.write_text(json.dumps(record, indent=2) + "\n")
        else:
            for offset in (0, global_experts - local):
                expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device="cuda")
                expert_map[offset : offset + local] = torch.arange(local, dtype=torch.int32, device="cuda")
                outside = local if offset == 0 else 0
                routes = {
                    "all-local": (all_rows + slots) % local + offset,
                    "mixed": torch.where(
                        slots < 4, (all_rows + slots) % local + offset, (all_rows + slots) % 24 + outside
                    ),
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
                        row["reference_passed"] = torch.allclose(before.float(), expected.float(), atol=0.02, rtol=0.02)
                        run(inputs.flip(0).contiguous(), routes["mixed"].flip(0).contiguous(), probabilities)
                        after = run(x, target, weights)
                        row["history_max_abs"] = (before.float() - after.float()).abs().max().item()
                        row["history_passed"] = torch.equal(before, after)
                        row["filtering_passed"] = True
                        if name == "no-local":
                            row["filtering_passed"] = not torch.count_nonzero(before).item()
                        if name == "negative-padding":
                            row["filtering_passed"] = not torch.count_nonzero(
                                before[torch.arange(count, device="cuda") % 3 == 0]
                            ).item()
                        row["status"] = (
                            "passed"
                            if row["reference_passed"] and row["history_passed"] and row["filtering_passed"]
                            else "failed"
                        )
                        output.write_text(json.dumps(record, indent=2) + "\n")
            assert any(x["output_max_abs"] > 0.01 for x in record["cases"])
            record["completed_phases"] = ["ep-history"]
            output.write_text(json.dumps(record, indent=2) + "\n")
            # Same persistent native intermediate buffers across differently sized
            # CUDA graphs; graph-private alignment/output allocations stay separate.
            record["cuda_graph_history"] = []
            graphs = []
            for count in (1, 128, 2048):
                x, target, weights = inputs[:count], routes["mixed"][:count], probabilities[:count]
                run(x, target, weights)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = run(x, target, weights)
                graphs.append((count, graph, result))
            reverse_inputs = inputs.flip(0).contiguous()
            reverse_ids = routes["mixed"].flip(0).contiguous()
            run(reverse_inputs, reverse_ids, probabilities)
            torch.cuda.synchronize()
            other = torch.cuda.CUDAGraph()
            with torch.cuda.graph(other):
                run(reverse_inputs, reverse_ids, probabilities)
            for count, graph, result in graphs:
                graph.replay()
                torch.cuda.synchronize()
                before = result.clone()
                other.replay()
                torch.cuda.synchronize()
                cache13.fill_(16)
                cache2.fill_(-16)
                out.fill_(32)
                graph.replay()
                torch.cuda.synchronize()
                row = {"tokens": count, "maximum": (before.float() - result.float()).abs().max().item()}
                record["cuda_graph_history"].append(row)
                output.write_text(json.dumps(record, indent=2) + "\n")
                row["status"] = "passed" if torch.equal(before, result) else "failed"
                output.write_text(json.dumps(record, indent=2) + "\n")
            record["completed_phases"].append("cuda-graph-history")
            output.write_text(json.dumps(record, indent=2) + "\n")
        # Two legal full-region orders isolate physical routed-token layout.
        # Repeated identical orders and canonical orders get fresh zeroed
        # output/intermediates/workspace to separate history from layout.
        for mode in ("exact-representable", "random-bf16-source"):
            if mode == "random-bf16-source":
                magnitude = 2.0 ** (torch.arange(hidden, device="cuda") % 9 - 4)
                w13 = (torch.randn_like(w13) * magnitude / math.sqrt(hidden)).bfloat16()
                w2 = (torch.randn_like(w2) * magnitude / math.sqrt(intermediate)).bfloat16()
                q13, s13 = quantize(w13, exact=False)
                q2, s2 = quantize(w2, exact=False)
                layer.w13_weight = w13
                q13, q2, s13, s2 = prepare_fp8_moe_layer_for_marlin(layer, q13, q2, s13, s2)
            expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device="cuda")
            expert_map[:local] = torch.arange(local, dtype=torch.int32, device="cuda")
            for count in (16, 115, 128, 129, 256, 512, 2048):
                for route in ("hot-local-eight", "hot-local-four-mixed"):
                    ids = torch.arange(topk, device="cuda").expand(count, -1).contiguous()
                    if route == "hot-local-four-mixed":
                        ids[:, 4:] += local
                    x = (inputs[:count] * (2.0 ** (torch.arange(hidden, device="cuda") % 9 - 4))).bfloat16()
                    weights = probabilities[:count]
                    estimated = math.ceil(count * local / global_experts)
                    for block in (8, 16, 32, 48, 64):
                        if estimated * topk / local / block < 0.9:
                            break
                    original, experts, padded = moe_align_block_size(
                        ids, block, global_experts, expert_map, ignore_invalid_experts=True
                    )
                    ascending, descending, used, regions = equivalent_layouts(
                        original, experts, padded, block, ids.numel()
                    )
                    canonical_a = _canonicalize_marlin_moe_token_order(ascending, experts, padded, block, ids.numel())
                    canonical_b = _canonicalize_marlin_moe_token_order(descending, experts, padded, block, ids.numel())
                    assert torch.equal(canonical_a[:used], canonical_b[:used])
                    assert torch.equal(canonical_a[:used], ascending[:used])
                    args = (x, ids, weights, expert_map, q13, q2, s13, s2, layer)
                    first = layout_outputs(*args, ascending, experts, padded, block)
                    same = layout_outputs(*args, ascending, experts, padded, block)
                    different = layout_outputs(*args, descending, experts, padded, block)
                    canonical_first = layout_outputs(*args, canonical_a, experts, padded, block)
                    canonical_second = layout_outputs(*args, canonical_a, experts, padded, block)
                    row = {
                        "weight_mode": mode,
                        "routing": route,
                        "tokens": count,
                        "block_size": block,
                        "used_slots": used,
                        "expert_regions": regions,
                        "layouts_differ": not torch.equal(ascending[:used], descending[:used]),
                        "canonical_active_layouts_identical": True,
                        "same_order_w13_max": maximum_difference(first[0], same[0]),
                        "same_order_output_max": maximum_difference(first[1], same[1]),
                        "reversed_order_w13_max": maximum_difference(first[0], different[0]),
                        "reversed_order_output_max": maximum_difference(first[1], different[1]),
                        "canonical_repeat_w13_max": maximum_difference(canonical_first[0], canonical_second[0]),
                        "canonical_repeat_output_max": maximum_difference(canonical_first[1], canonical_second[1]),
                    }
                    record["layout_order"].append(row)
                    output.write_text(json.dumps(record, indent=2) + "\n")
    record |= {
        "status": "completed-diagnostic",
        "finished_unix": time.time(),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "history_failures": None if reused else sum(x["status"] != "passed" for x in record["cases"]),
        "graph_history_failures": None if reused else sum(x["status"] != "passed" for x in record["cuda_graph_history"]),
        "order_sensitive_cases": sum(
            x["reversed_order_w13_max"] > 0 or x["reversed_order_output_max"] > 0 for x in record["layout_order"]
        ),
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": record["status"],
                "history_cases": len(record["cases"]),
                "order_cases": len(record["layout_order"]),
                "history_reused": bool(reused),
            }
        )
    )


if __name__ == "__main__":
    main()
