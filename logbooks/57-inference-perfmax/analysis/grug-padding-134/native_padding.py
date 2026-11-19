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

import torch
import vllm
import vllm._custom_ops as ops
import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.online.fp8 import _fp8_scale
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_moe_layer_for_marlin
from vllm.model_executor.models.grugmoe import GrugMoeRouter
from vllm.scalar_type import scalar_types


def quantize(weight):
    scales = _fp8_scale(weight.float().abs().flatten(1).amax(dim=1))
    packed = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    for expert in range(weight.shape[0]):
        packed[expert] = ops.scaled_fp8_quant(weight[expert], scale=scales[expert])[0]
    return packed, scales


def maximum_difference(first, second):
    return (first.float() - second.float()).abs().max().item()


def tensor_digest(value):
    raw = value.contiguous().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for offset in range(0, raw.numel(), 16 * 1024**2):
        digest.update(raw[offset : offset + 16 * 1024**2].cpu().numpy().tobytes())
    return digest.hexdigest()


def router_timings(config):
    rows = []
    router = GrugMoeRouter(8, 384, torch.zeros(384, device="cuda"))
    for count in (4, 128, 2048):
        hidden = torch.ones((count, 64), device="cuda", dtype=torch.bfloat16)
        logits = torch.randn((count, 384), device="cuda")
        padding = torch.arange(count, device="cuda") >= max(1, count // 2)
        with set_forward_context(None, config, num_tokens=count, is_padding=padding):
            for _ in range(10):
                router.select_experts(hidden_states=hidden, router_logits=logits)
            measured = []
            for _ in range(3):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100):
                    router.select_experts(hidden_states=hidden, router_logits=logits)
                end.record()
                end.synchronize()
                measured.append(start.elapsed_time(end) * 1000 / 100)
            rows.append({"physical_rows": count, "microseconds_per_call": measured})
    return rows


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
    assert importlib.metadata.version("vllm") == payload["candidate_manifest"]["version"]
    assert torch.__version__ == "2.14.1+cu132"
    assert envs.VLLM_MOE_SKIP_PADDING
    expect_stable = payload.get("expect_stable_padding_ids", False)
    assert not Path(vllm.__file__).resolve().is_relative_to(Path(payload["source_root"]).resolve())
    binary_hashes = {}
    for name in ("vllm/_C_stable_libtorch.abi3.so", "vllm/_moe_C_stable_libtorch.abi3.so"):
        path = Path(vllm.__file__).resolve().parent.parent / name
        with path.open("rb") as stream:
            binary_hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        assert binary_hashes[name] == payload["native_members"][name]
    record = {
        "scope": (
            "One-GPU component: current-main Grug router and native Marlin FP8; raw CUDA graphs. "
            "No canonical ordering, selective-FP8 or hybrid patches; no full-model attribution."
        ),
        "status": "running",
        "source_commit": payload["source_commit"],
        "research_source": payload["research_source"],
        "wheel_sha256": payload["candidate_manifest"]["wheels"][0]["sha256"],
        "marlin_sha256": payload["marlin_sha256"],
        "model_sha256": payload["model_sha256"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm_package": str(Path(vllm.__file__).resolve()),
        "native_binary_sha256": binary_hashes,
        "dependency_versions": {
            distribution.metadata["Name"]: distribution.version for distribution in importlib.metadata.distributions()
        },
        "marlin_module": str(
            Path(importlib.import_module("vllm.model_executor.layers.fused_moe.experts.marlin_moe").__file__).resolve()
        ),
        "base_commit": payload["compiled_base"],
        "excluded_patches": ["#133 canonical ordering", "#132 selective FP8", "hybrid backend selection"],
        "device": torch.cuda.get_device_name(),
        "moe_skip_padding": envs.VLLM_MOE_SKIP_PADDING,
        "expected_stable_padding_ids": expect_stable,
        "router_timing_scope": (
            "CUDA events across eager layer calls; includes dispatch gaps; no serving throughput claim"
        ),
        "started_unix": time.time(),
        "weight_inputs": [],
        "cases": [],
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
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
            record["weight_inputs"].append(
                {
                    "local_experts": local,
                    "seed": 42,
                    "sha256": {
                        name: tensor_digest(value)
                        for name, value in (("inputs", inputs), ("q13", q13), ("q2", q2), ("s13", s13), ("s2", s2))
                    },
                }
            )
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
                original_logits = logits.clone()
                original_choices = {name: value.clone() for name, value in choices.items()}

                with set_forward_context(None, config, num_tokens=count, is_padding=padding):
                    routed.copy_(source)
                    base_weights, base_ids = router.select_experts(hidden_states=routed, router_logits=logits)
                    assert torch.equal(routed[:real], source[:real])
                    assert not torch.count_nonzero(routed[real:]).item()
                    assert not torch.count_nonzero(base_weights[real:]).item()
                    first, same = fresh(logits), fresh(logits)
                    assert torch.isfinite(first).all() and torch.isfinite(same).all()
                    assert first[:real].abs().max().item() > 0
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
                        assert ((other_ids >= 0) & (other_ids < global_experts)).all()
                        changed, restored = fresh(alternate), fresh(logits)
                        assert torch.isfinite(changed).all()
                        assert torch.equal(logits, original_logits)
                        assert torch.equal(alternate, original_choices[name])
                        row = {
                            "local_experts": local,
                            "physical_rows": count,
                            "real_rows": real,
                            "padded_route": name,
                            "padded_ids_differ": padded_ids_differ,
                            "real_input_and_routes_identical": True,
                            "caller_logits_unchanged": True,
                            "padded_hidden_and_weights_zero": True,
                            "same_route_max": maximum_difference(first[:real], same[:real]),
                            "changed_padded_route_real_max": maximum_difference(first[:real], changed[:real]),
                            "restored_route_max": maximum_difference(first[:real], restored[:real]),
                            "padded_output_max": changed[real:].float().abs().max().item(),
                        }
                        repeat_case = (local, count, real, name) in (
                            (24, 4, 1, "other-local"),
                            (24, 128, 1, "non-local"),
                            (96, 16, 1, "other-local"),
                        )
                        if repeat_case:
                            row["repeat_trials"] = []
                            for index in range(payload.get("repeats", 0)):
                                a, b = (logits, alternate) if index % 2 == 0 else (alternate, logits)
                                reference, repeated, different, restored = fresh(a), fresh(a), fresh(b), fresh(a)
                                row["repeat_trials"].append(
                                    {
                                        "direction": "A-to-B" if index % 2 == 0 else "B-to-A",
                                        "same": maximum_difference(reference[:real], repeated[:real]),
                                        "changed": maximum_difference(reference[:real], different[:real]),
                                        "restored": maximum_difference(reference[:real], restored[:real]),
                                    }
                                )
                        record["cases"].append(row)
                        output.write_text(json.dumps(record, indent=2) + "\n")
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
                            row["graph_same_route_max"] = maximum_difference(graph_first[:real], captured[:real])
                            active_logits.copy_(alternate)
                            graph.replay()
                            torch.cuda.synchronize()
                            row["graph_changed_padded_route_real_max"] = maximum_difference(
                                graph_first[:real], captured[:real]
                            )
                            active_logits.copy_(logits)
                            graph.replay()
                            torch.cuda.synchronize()
                            row["graph_restored_route_max"] = maximum_difference(graph_first[:real], captured[:real])
                            if repeat_case:
                                row["graph_repeat_trials"] = []
                                for index in range(payload.get("repeats", 0)):
                                    a, b = (logits, alternate) if index % 2 == 0 else (alternate, logits)
                                    active_logits.copy_(a)
                                    graph.replay()
                                    torch.cuda.synchronize()
                                    reference = captured[:real].clone()
                                    graph.replay()
                                    torch.cuda.synchronize()
                                    repeated = maximum_difference(reference, captured[:real])
                                    active_logits.copy_(b)
                                    graph.replay()
                                    torch.cuda.synchronize()
                                    different = maximum_difference(reference, captured[:real])
                                    active_logits.copy_(a)
                                    graph.replay()
                                    torch.cuda.synchronize()
                                    row["graph_repeat_trials"].append(
                                        {
                                            "direction": "A-to-B" if index % 2 == 0 else "B-to-A",
                                            "same": repeated,
                                            "changed": different,
                                            "restored": maximum_difference(reference, captured[:real]),
                                        }
                                    )
                        output.write_text(json.dumps(record, indent=2) + "\n")
        record["router_timing"] = router_timings(config)
    record |= {
        "status": "completed-diagnostic",
        "finished_unix": time.time(),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "padded_route_sensitive_cases": sum(row["changed_padded_route_real_max"] > 0 for row in record["cases"]),
        "same_route_failures": sum(row["same_route_max"] > 0 for row in record["cases"]),
        "restored_route_failures": sum(row["restored_route_max"] > 0 for row in record["cases"]),
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
                    payload["pytest_selection"],
                    "-v",
                    "--maxfail=1",
                    "--junitxml=" + str(output.with_suffix(".xml")),
                ],
            )
        )
        output.write_text(json.dumps(record, indent=2) + "\n")
        assert record["pytest_exit"] == 0
    print(json.dumps({k: record[k] for k in ("status", "padded_route_sensitive_cases", "same_route_failures")}))


if __name__ == "__main__":
    main()
