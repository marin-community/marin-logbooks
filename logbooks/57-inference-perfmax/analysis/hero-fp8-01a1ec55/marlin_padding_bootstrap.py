# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Vary only zero-weight padding routes through the installed Grug/Marlin path."""

import hashlib
import importlib
import importlib.metadata
import json
import math
import runpy
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import marlin_history_bootstrap as control
import torch
import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    _canonicalize_marlin_moe_token_order,
    fused_marlin_moe,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_moe_layer_for_marlin
from vllm.model_executor.models.grugmoe import GrugMoeRouter
from vllm.scalar_type import scalar_types


def run_gating(gating, *, source, routed, router, packed, expert_map, workspace, caches, output):
    routed.copy_(source)
    weights, ids = router.select_experts(hidden_states=routed, router_logits=gating)
    q13, q2, s13, s2 = packed
    cache13, cache2 = caches
    return fused_marlin_moe(
        routed,
        q13,
        q2,
        None,
        None,
        s13,
        s2,
        weights,
        ids,
        scalar_types.float8_e4m3fn.id,
        global_num_experts=expert_map.numel(),
        expert_map=expert_map,
        workspace=workspace,
        intermediate_cache13=cache13,
        intermediate_cache2=cache2,
        output=output[: len(source)],
    ).clone()


def fresh_gating(gating, *, run, buffers):
    for buffer in buffers:
        buffer.zero_()
    return run(gating)


def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    output = Path(payload["output"])
    for name, expected in (
        ("vllm.model_executor.layers.fused_moe.experts.marlin_moe", payload["marlin_sha256"]),
        ("vllm.model_executor.models.grugmoe", payload["model_sha256"]),
    ):
        module = importlib.import_module(name)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected
    assert hashlib.sha256(Path(control.__file__).read_bytes()).hexdigest() == payload["control_sha256"]
    assert importlib.metadata.version("vllm") == payload["candidate_manifest"]["version"]
    assert torch.__version__ == "2.14.1+cu132"
    assert envs.VLLM_MOE_SKIP_PADDING
    expect_stable = payload.get("expect_stable_padding_ids", False)
    record = {
        "scope": (
            "One-GPU component: actual Grug router, canonical Marlin FP8 and raw CUDA graphs; no full-model attribution"
        ),
        "status": "running",
        "source_commit": payload["source_commit"],
        "research_source": payload["research_source"],
        "wheel_sha256": payload["candidate_manifest"]["wheels"][0]["sha256"],
        "marlin_sha256": payload["marlin_sha256"],
        "model_sha256": payload["model_sha256"],
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "moe_skip_padding": envs.VLLM_MOE_SKIP_PADDING,
        "expected_stable_padding_ids": expect_stable,
        "started_unix": time.time(),
        "cases": [],
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
    if payload.get("run_focused_pytest"):
        root = Path(payload["source_root"])
        sys.path.insert(0, str(root / "infra/release"))
        native = runpy.run_path(str(root / "infra/release/gpu_validation.py"))
        record["pytest_exit"] = native["run_wheel_tests"](
            SimpleNamespace(
                package_source_root=root,
                validation_source_root=root,
                exclude_test=[],
                pytest_args=[
                    str(root / "tests/models/test_grugmoe.py"),
                    "-k",
                    "online_fp8_loading_matches_vllm or hero_online_fp8_latent_and_shared "
                    "or quantization_platform_support or preserves_gpu_tensor_parallel_restriction "
                    "or router_padding_ids_ignore_stale_logits",
                    "-v",
                    "--maxfail=1",
                    "--junitxml=" + str(output.with_suffix(".xml")),
                ],
            )
        )
        output.write_text(json.dumps(record, indent=2) + "\n")
        assert record["pytest_exit"] == 0
    global_experts, hidden, intermediate, topk, maximum = 384, 3072, 3072, 8, 2048
    config = VllmConfig()
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode(), set_current_vllm_config(config):
        for local in (24, 96):
            torch.manual_seed(42)
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
            expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device="cuda")
            expert_map[:local] = torch.arange(local, dtype=torch.int32, device="cuda")
            router = GrugMoeRouter(topk, global_experts, torch.zeros(global_experts, device="cuda"))
            cache13 = torch.empty(maximum * topk * 2 * intermediate, dtype=torch.bfloat16, device="cuda")
            cache2 = torch.empty(maximum * topk * intermediate, dtype=torch.bfloat16, device="cuda")
            out = torch.empty((maximum, hidden), dtype=torch.bfloat16, device="cuda")
            for count, real in ((4, 1), (8, 1), (16, 1), (128, 1), (128, 115), (512, 129), (2048, 129)):
                source = torch.full((count, hidden), 32, device="cuda", dtype=torch.bfloat16)
                source[:real] = (inputs[:real] * magnitude).bfloat16()
                routed = source.clone()
                padding = torch.arange(count, device="cuda") >= real
                logits = torch.full((count, global_experts), -8.0, device="cuda")
                logits[:, :topk] = torch.arange(topk, device="cuda") + 1.0
                choices = {"same-experts": logits.clone(), "other-local": logits.clone(), "non-local": logits.clone()}
                for name, start in (("other-local", 16), ("non-local", local + 8)):
                    choices[name][real:].fill_(-8)
                    choices[name][real:, start : start + topk] = torch.arange(topk, device="cuda") + 1.0

                run = partial(
                    run_gating,
                    source=source,
                    routed=routed,
                    router=router,
                    packed=(q13, q2, s13, s2),
                    expert_map=expert_map,
                    workspace=layer.workspace,
                    caches=(cache13, cache2),
                    output=out,
                )
                fresh = partial(fresh_gating, run=run, buffers=(layer.workspace, cache13, cache2, out))

                with set_forward_context(None, config, num_tokens=count, is_padding=padding):
                    routed.copy_(source)
                    base_weights, base_ids = router.select_experts(hidden_states=routed, router_logits=logits)
                    assert torch.equal(routed[:real], source[:real])
                    assert not torch.count_nonzero(routed[real:]).item()
                    assert not torch.count_nonzero(base_weights[real:]).item()
                    first, same = fresh(logits), fresh(logits)
                    for name in ("other-local", "non-local"):
                        alternate = choices[name]
                        routed.copy_(source)
                        other_weights, other_ids = router.select_experts(hidden_states=routed, router_logits=alternate)
                        assert torch.equal(routed[:real], source[:real])
                        assert not torch.count_nonzero(routed[real:]).item()
                        assert torch.equal(base_ids[:real], other_ids[:real])
                        assert torch.equal(base_weights, other_weights)
                        padded_ids_differ = not torch.equal(base_ids[real:], other_ids[real:])
                        assert padded_ids_differ != expect_stable
                        changed, restored = fresh(alternate), fresh(logits)
                        row = {
                            "local_experts": local,
                            "physical_rows": count,
                            "real_rows": real,
                            "padded_route": name,
                            "padded_ids_differ": padded_ids_differ,
                            "real_input_and_routes_identical": True,
                            "padded_hidden_and_weights_zero": True,
                            "same_route_max": control.maximum_difference(first[:real], same[:real]),
                            "changed_padded_route_real_max": control.maximum_difference(first[:real], changed[:real]),
                            "restored_route_max": control.maximum_difference(first[:real], restored[:real]),
                            "padded_output_max": changed[real:].float().abs().max().item(),
                        }
                        record["cases"].append(row)
                        output.write_text(json.dumps(record, indent=2) + "\n")
                        if row["changed_padded_route_real_max"] > 0:
                            estimated = math.ceil(count * local / global_experts)
                            for block in (8, 16, 32, 48, 64):
                                if estimated * topk / local / block < 0.9:
                                    break
                            intermediate_results = []
                            for ids in (base_ids, other_ids):
                                order, experts, padded = moe_align_block_size(
                                    ids, block, global_experts, expert_map, ignore_invalid_experts=True
                                )
                                order = _canonicalize_marlin_moe_token_order(order, experts, padded, block, ids.numel())
                                w13_result, contributions = control.layout_outputs(
                                    routed,
                                    ids,
                                    base_weights,
                                    expert_map,
                                    q13,
                                    q2,
                                    s13,
                                    s2,
                                    layer,
                                    order,
                                    experts,
                                    padded,
                                    block,
                                )
                                intermediate_results.append(
                                    (w13_result.view(count, topk, -1)[:real], contributions.view(count, topk, -1)[:real])
                                )
                            row["separate_buffer_real_w13_max"] = control.maximum_difference(
                                intermediate_results[0][0], intermediate_results[1][0]
                            )
                            row["separate_buffer_real_contribution_max"] = control.maximum_difference(
                                intermediate_results[0][1], intermediate_results[1][1]
                            )
                        if (count, real) in ((16, 1), (128, 1), (128, 115), (512, 129)):
                            active_logits = logits.clone()
                            run(active_logits)
                            torch.cuda.synchronize()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                captured = run(active_logits)
                            graph.replay()
                            torch.cuda.synchronize()
                            graph_first = captured.clone()
                            graph.replay()
                            torch.cuda.synchronize()
                            row["graph_same_route_max"] = control.maximum_difference(graph_first[:real], captured[:real])
                            active_logits.copy_(alternate)
                            graph.replay()
                            torch.cuda.synchronize()
                            row["graph_changed_padded_route_real_max"] = control.maximum_difference(
                                graph_first[:real], captured[:real]
                            )
                            active_logits.copy_(logits)
                            graph.replay()
                            torch.cuda.synchronize()
                            row["graph_restored_route_max"] = control.maximum_difference(
                                graph_first[:real], captured[:real]
                            )
                        output.write_text(json.dumps(record, indent=2) + "\n")
    record |= {
        "status": "completed-diagnostic",
        "finished_unix": time.time(),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "padded_route_sensitive_cases": sum(row["changed_padded_route_real_max"] > 0 for row in record["cases"]),
        "same_route_failures": sum(row["same_route_max"] > 0 for row in record["cases"]),
        "restored_route_failures": sum(row["restored_route_max"] > 0 for row in record["cases"]),
    }
    if expect_stable:
        failed = any(
            row[key] != 0
            for row in record["cases"]
            for key in (
                "same_route_max",
                "changed_padded_route_real_max",
                "restored_route_max",
                "padded_output_max",
                "graph_same_route_max",
                "graph_changed_padded_route_real_max",
                "graph_restored_route_max",
            )
            if key in row
        )
        record["status"] = "failed" if failed else "passed"
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("status", "padded_route_sensitive_cases", "same_route_failures")}))
    if expect_stable:
        assert record["status"] == "passed"


if __name__ == "__main__":
    main()
