# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify native piecewise graphs on one otherwise idle allocated GPU.

Use the same retained synthetic Hero checkpoint for NONE and PIECEWISE in
separate processes. Never remove or monkeypatch a model's unsupported-mode
guard. Record changing-batch, shared-prefix and branch outputs before comparing.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from native_shortconv_probe import install_forward
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.model_executor.models import grugmoe


def pad_reference_dispatch(result, capture_sizes: list[int]):
    """Match PIECEWISE's cross-DP row counts while executing with graphs off.

    Only a numerical reference uses this. Native DP coordination runs first;
    model inputs, attention/state metadata and supported-mode guards stay native.
    """
    descriptor, sync = result
    assert descriptor.cg_mode == CUDAGraphMode.NONE
    assert descriptor.num_ubatches == 1
    if sync is None:
        return result
    largest = int(sync.num_tokens_across_dp.max())
    if largest == 0 or largest > max(capture_sizes):
        return result
    padded = min(size for size in capture_sizes if size >= largest)
    return (
        replace(descriptor, num_tokens=padded),
        replace(sync, num_tokens_across_dp=torch.full_like(sync.num_tokens_across_dp, padded)),
    )


def install_batch_trace(
    worker, reference_capture_sizes: list[int] | None = None, disable_graph_replay: bool = False
) -> None:
    # Record native CPU dispatch metadata; do not synchronize or profile the GPU.
    worker.hero_batch_trace = []
    module = sys.modules[type(worker.model_runner).__module__]
    dispatch = module.dispatch_cg_and_sync_dp
    worker.hero_dispatch_original = dispatch

    def traced_dispatch(*args, **kwargs):
        result = dispatch(*args, **kwargs)
        if reference_capture_sizes:
            result = pad_reference_dispatch(result, reference_capture_sizes)
        if disable_graph_replay:
            descriptor, sync = result
            assert descriptor.cg_mode in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE)
            result = (replace(descriptor, cg_mode=CUDAGraphMode.NONE), sync)
        descriptor = result[0]
        sync = result[1]
        worker.hero_batch_trace.append(
            {
                "requests": args[1],
                "scheduled_tokens": args[2],
                "padded_tokens": descriptor.num_tokens,
                "graph_mode": descriptor.cg_mode.name,
                "max_query_len": kwargs.get("max_query_len"),
                "tokens_across_dp": sync.num_tokens_across_dp.tolist() if sync is not None else None,
                "reference_capture_sizes": reference_capture_sizes,
                "disable_graph_replay": disable_graph_replay,
            }
        )
        return result

    module.dispatch_cg_and_sync_dp = traced_dispatch


def take_batch_trace(worker) -> list[dict]:
    rows = worker.hero_batch_trace
    worker.hero_batch_trace = []
    return rows


def stop_batch_trace(worker) -> None:
    module = sys.modules[type(worker.model_runner).__module__]
    module.dispatch_cg_and_sync_dp = worker.hero_dispatch_original


def cache_storage(worker) -> dict:
    """Record native CUDA KV tensor storage metadata without reading GPU data."""
    runner = worker.model_runner
    storage = {}
    for tensor in runner.kv_caches:
        assert tensor.is_cuda
        allocation = tensor.untyped_storage()
        key = (str(tensor.device), allocation.data_ptr())
        storage[key] = {"bytes": allocation.nbytes(), "backing_tensor_dtype": str(tensor.dtype)}
    record = {
        "unix": time.time(),
        "dp_rank": runner.dp_rank,
        "native_num_blocks": runner.kv_cache_config.num_blocks,
        "cache_tensor_count": len(runner.kv_caches),
        "unique_cuda_storage_count": len(storage),
        "unique_cuda_storage_bytes": sum(item["bytes"] for item in storage.values()),
        "storage": list(storage.values()),
        "scope": "KV tensor storage only; excludes allocator reservation and graph buffers. CPU metadata, no GPU sync.",
    }
    path = Path(os.environ["IRIS_OUTPUT_DIR"]) / f"kv-storage-dp{runner.dp_rank}.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return record


def set_shortconv_reference(worker, source: str = "", sha256: str = "") -> dict:
    """Select a diagnostic implementation outside the captured graph segments."""
    config = worker.vllm_config.compilation_config
    assert config.cudagraph_mode == CUDAGraphMode.PIECEWISE
    assert "vllm::grug_moe_short_conv" in config.splitting_ops
    assert hashlib.sha256(Path(grugmoe.__file__).read_bytes()).hexdigest() == (
        "78563576172625505820fb2aa2a19b4ee8e8f112d6435cd2bdafb6db311311b8"
    )
    if not hasattr(worker, "hero_shortconv_original"):
        worker.hero_shortconv_original = grugmoe.GrugMoeShortConv.forward_impl
    if source:
        install_forward(Path(source), sha256)
    else:
        grugmoe.GrugMoeShortConv.forward_impl = worker.hero_shortconv_original
    record = {"unix": time.time(), "dp_rank": worker.parallel_config.data_parallel_rank, "source_sha256": sha256}
    path = Path(os.environ["IRIS_OUTPUT_DIR"]) / f"shortconv-reference-dp{record['dp_rank']}.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return record


class BatchTraceWorkerExtension:
    install_hero_batch_trace = install_batch_trace
    take_hero_batch_trace = take_batch_trace
    stop_hero_batch_trace = stop_batch_trace
    hero_cache_storage = cache_storage
    set_hero_shortconv_reference = set_shortconv_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("NONE", "PIECEWISE"), required=True)
    parser.add_argument("--capture-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--deterministic-batches", action="store_true")
    parser.add_argument("--trace-batches", action="store_true")
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compilation = {
        "mode": 3,
        "cudagraph_mode": args.mode,
        "cudagraph_capture_sizes": args.capture_sizes,
        "splitting_ops": [
            *CompilationConfig._attention_ops,
            "vllm::unified_kv_cache_update",
            "vllm::unified_mla_kv_cache_update",
            "vllm::grug_moe_short_conv",
        ],
    }
    proof = {
        "scope": "One GB200, tiny Hero, TP1; not full-Hero, EP16, H100, or throughput qualification",
        "mode": args.mode,
        "prefix_caching": args.prefix_cache,
        "deterministic_batches": args.deterministic_batches,
        "compilation_requested": compilation,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in ("vllm", "torch", "flashinfer-python")},
        "rows": [],
    }
    started = time.monotonic()
    llm = LLM(
        model=str(args.model),
        worker_extension_cls="tiny_graph_probe.BatchTraceWorkerExtension" if args.trace_batches else "",
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=128,
        max_num_seqs=8,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.15,
        enforce_eager=False,
        enable_prefix_caching=args.prefix_cache,
        kernel_config={"moe_backend": "triton"},
        compilation_config=compilation,
        seed=42,
    )
    try:
        actual = llm.llm_engine.vllm_config
        proof.update(
            startup_seconds=time.monotonic() - started,
            actual_cudagraph_mode=actual.compilation_config.cudagraph_mode.name,
            actual_capture_sizes=actual.compilation_config.cudagraph_capture_sizes,
            use_v2_model_runner=actual.use_v2_model_runner,
        )
        assert proof["actual_cudagraph_mode"] == args.mode
        if args.trace_batches:
            llm.collective_rpc("install_hero_batch_trace")
        prefix = [1, 2, 3] * 16
        prompts = [prefix + suffix for suffix in ([1, 2, 3], [3, 2], [4, 5, 6])]
        for branches in ([0], [1, 0], [2, 0, 1], [0]) * 2:
            started = time.monotonic()
            inputs = [{"prompt_token_ids": prompts[index]} for index in branches]
            sampling = SamplingParams(temperature=0, seed=42, max_tokens=8, min_tokens=8, ignore_eos=True, logprobs=5)
            if args.deterministic_batches:
                # This barrier is only a numerical diagnostic, never an AgentX control.
                llm.llm_engine.engine_core.call_utility("pause_scheduler", "keep", False)
                try:
                    llm.enqueue(inputs, sampling)
                finally:
                    llm.llm_engine.engine_core.call_utility("resume_scheduler")
                outputs = llm.wait_for_completion(use_tqdm=False)
            else:
                outputs = llm.generate(inputs, sampling, use_tqdm=False)
            row = {
                "branches": branches,
                "seconds": time.monotonic() - started,
                "tokens": [list(output.outputs[0].token_ids) for output in outputs],
                "logprobs": [
                    [{str(key): value.logprob for key, value in item.items()} for item in output.outputs[0].logprobs]
                    for output in outputs
                ],
            }
            if args.trace_batches:
                row["internal_batches"] = llm.collective_rpc("take_hero_batch_trace")
            proof["rows"].append(row)
            assert all(
                math.isfinite(value) for request in row["logprobs"] for item in request for value in item.values()
            )
        proof["status"] = "completed_finite"
    finally:
        llm.llm_engine.engine_core.shutdown()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
