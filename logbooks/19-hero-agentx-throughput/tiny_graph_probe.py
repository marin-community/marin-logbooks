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
from local_piecewise_dp import local_piecewise_dispatch, validate_local_piecewise_config
from native_shortconv_probe import install_forward
from torch._inductor.codecache import PyCodeCache
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._inductor.select_algorithm import extern_kernels
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
    worker,
    reference_capture_sizes: list[int] | None = None,
    disable_graph_replay: bool = False,
    local_piecewise_dp: bool = False,
    checkpoint_first_dispatch: bool = False,
) -> None:
    # Record native CPU dispatch metadata; do not synchronize or profile the GPU.
    worker.hero_batch_trace = []
    checkpoint_written = False
    if local_piecewise_dp:
        assert reference_capture_sizes is None
        validate_local_piecewise_config(worker)
    module = sys.modules[type(worker.model_runner).__module__]
    dispatch = module.dispatch_cg_and_sync_dp
    worker.hero_dispatch_original = dispatch

    def traced_dispatch(*args, **kwargs):
        nonlocal checkpoint_written
        result = dispatch(*args, **kwargs)
        if local_piecewise_dp:
            result = local_piecewise_dispatch(result, args, kwargs)
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
                "local_piecewise_dp": local_piecewise_dp,
            }
        )
        if checkpoint_first_dispatch and not checkpoint_written:
            path = Path(os.environ["IRIS_OUTPUT_DIR"]) / (
                f"first-dispatch-local{local_piecewise_dp}-disabled{disable_graph_replay}"
                f"-dp{worker.model_runner.dp_rank}.json"
            )
            path.write_text(json.dumps({"unix": time.time(), "row": worker.hero_batch_trace[-1]}) + "\n")
            checkpoint_written = True
        return result

    module.dispatch_cg_and_sync_dp = traced_dispatch


def take_batch_trace(worker) -> list[dict]:
    rows = worker.hero_batch_trace
    path = Path(os.environ["IRIS_OUTPUT_DIR"]) / f"dispatch-traces-dp{worker.model_runner.dp_rank}.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps({"unix": time.time(), "rows": rows}) + "\n")
    worker.hero_batch_trace = []
    return rows


def stop_batch_trace(worker) -> None:
    module = sys.modules[type(worker.model_runner).__module__]
    module.dispatch_cg_and_sync_dp = worker.hero_dispatch_original


def capture_compiler_sources(worker, label: str) -> dict:
    """Keep emitted code and tuning settings before numerical or timing phases."""
    rank = worker.model_runner.dp_rank
    local_size = worker.vllm_config.parallel_config.data_parallel_size_local
    if rank % local_size:
        return {"dp_rank": rank, "capturing": False}
    assert label in ("ready", "after-numerics")
    destination = Path(os.environ["IRIS_OUTPUT_DIR"]) / "compiler-sources" / label
    destination.mkdir(parents=True, exist_ok=False)
    files = []
    total = 0
    for kind, key in [("vllm", "VLLM_CACHE_ROOT"), ("inductor", "TORCHINDUCTOR_CACHE_DIR")]:
        root = Path(os.environ[key])
        for source in root.rglob("*"):
            if not source.is_file():
                continue
            selected = source.name in ("computation_graph.py", "autotune_configs.json")
            if kind == "inductor":
                selected = source.suffix in (".py", ".cpp", ".ptx", ".ttir", ".ttgir", ".best_config")
            if not selected:
                continue
            size = source.stat().st_size
            assert size <= 16_000_000
            total += size
            assert total <= 256_000_000
            data = source.read_bytes()
            assert len(data) == size == source.stat().st_size
            target = destination / kind / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            files.append(
                {
                    "source": str(source),
                    "retained": str(target),
                    "bytes": size,
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    settings = {
        "inductor_max_autotune": os.environ.get("VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE", "1"),
        "inductor_coordinate_descent": os.environ.get("VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING", "1"),
        "flashinfer_autotune": worker.vllm_config.kernel_config.enable_flashinfer_autotune,
    }
    runtime_kernels = []
    for module in PyCodeCache.modules:
        for name, value in vars(module).items():
            if isinstance(value, CachingAutotuner):
                runtime_kernels.append(
                    {
                        "module_file": module.__file__,
                        "name": name,
                        "launchers": [
                            {
                                "kwargs": launcher.config.kwargs,
                                "num_warps": launcher.config.num_warps,
                                "num_stages": launcher.config.num_stages,
                                "cache_hash": launcher.cache_hash,
                                "n_regs": launcher.n_regs,
                                "n_spills": launcher.n_spills,
                            }
                            for launcher in value.launchers
                        ],
                    }
                )
    record = {
        "dp_rank": rank,
        "capturing": True,
        "label": label,
        "files": files,
        "bytes": total,
        "settings": settings,
        "runtime_kernels": runtime_kernels,
    }
    (destination / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def cache_storage(worker) -> dict:
    """Record native CUDA KV tensor storage metadata without reading GPU data."""
    runner = worker.model_runner
    config = worker.vllm_config.cache_config
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
        "cache_config": {
            "enable_prefix_caching": config.enable_prefix_caching,
            "prefix_cache_retention_interval": config.prefix_cache_retention_interval,
            "mamba_cache_mode": config.mamba_cache_mode,
            "mamba_block_size": config.mamba_block_size,
            "kv_cache_size_tokens": config.kv_cache_size_tokens,
        },
        "cache_groups": [
            {
                "group": index,
                "layers": len(group.layer_names),
                "spec_type": type(group.kv_cache_spec).__name__,
                "spec": str(group.kv_cache_spec),
            }
            for index, group in enumerate(runner.kv_cache_config.kv_cache_groups)
        ],
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


def install_moe_snapshot(worker, expected_layers: int = 4, row_limit: int | None = None) -> dict:
    """Retain prefill tensors, bounding full-Hero captures after its first layer."""
    assert (expected_layers, row_limit) in ((4, None), (48, 16))
    rank = worker.model_runner.dp_rank
    if rank != 0 and expected_layers == 4:
        return {"dp_rank": rank, "capturing": False}
    layers = worker.vllm_config.compilation_config.static_forward_context
    names = sorted(name for name in layers if name.endswith(".mlp.experts"))
    assert len(names) == expected_layers
    first_layer = names[0]
    if rank != 0:
        names = [first_layer]
    counts = dict.fromkeys(names, 0)
    for name in names:
        layer = layers[name]
        original = layer._forward_impl

        def snapshot(hidden, logits, shared_input, input_ids=None, *, _name=name, _original=original):
            prefill = (
                worker.hero_batch_trace[-1]["tokens_across_dp"][0] >= 16
                if expected_layers == 48
                else hidden.shape[0] >= 16
            )
            selected = counts[_name] < 2 and prefill
            if selected:
                rows = None if _name == first_layer else row_limit
                record = {
                    "layer": _name,
                    "hidden": hidden[:rows].detach().cpu(),
                    "router_logits": logits[:rows].detach().cpu(),
                    "hidden_stride": list(hidden.stride()),
                    "router_stride": list(logits.stride()),
                }
                if expected_layers == 48:
                    record.update(
                        dp_rank=rank,
                        full_hidden_shape=list(hidden.shape),
                        dispatch=worker.hero_batch_trace[-1].copy(),
                        real_http_rank=rank == 0,
                    )
            output = _original(hidden, logits, shared_input, input_ids)
            if selected:
                assert isinstance(output, torch.Tensor)
                record.update(output=output[:rows].detach().cpu(), output_stride=list(output.stride()))
                if expected_layers == 48:
                    after = hidden[:rows].detach().cpu()
                    record["hidden_input_unchanged"] = torch.equal(record["hidden"], after)
                    if not record["hidden_input_unchanged"]:
                        record["hidden_after"] = after
                    record["finite"] = {
                        key: bool(torch.isfinite(record[key]).all()) for key in ["hidden", "router_logits", "output"]
                    }
                prefix = f"moe-snapshot-dp{rank}" if expected_layers == 48 else "moe-snapshot"
                path = Path(os.environ["IRIS_OUTPUT_DIR"]) / f"{prefix}-{_name}-{counts[_name]}.pt"
                torch.save(record, path)
                counts[_name] += 1
            return output

        layer._forward_impl = snapshot
    return {"dp_rank": rank, "capturing": True, "layers": names, "max_snapshots_per_layer": 2}


def install_latent_projection_snapshot(worker) -> dict:
    """Observe two first-layer matmuls without replacing their arithmetic."""
    rank = worker.model_runner.dp_rank
    if rank != 0:
        return {"dp_rank": rank, "capturing": False}
    assert not hasattr(worker, "hero_latent_snapshot_count")
    modules = dict(worker.model_runner.get_model().named_modules())
    mlp = modules["model.layers.0.mlp"]
    weight = mlp.latent_down_proj.weight
    norm = mlp.latent_norm
    assert weight.ndim == 2 and weight.dtype == torch.bfloat16
    original = extern_kernels.mm
    worker.hero_latent_snapshot_count = 0

    def observed_mm(left, right, *, out):
        result = original(left, right, out=out)
        selected = (
            worker.hero_latent_snapshot_count < 2
            and left.shape[0] >= 16
            and right.data_ptr() == weight.data_ptr()
            and tuple(right.shape) == tuple(reversed(weight.shape))
        )
        if selected:
            index = worker.hero_latent_snapshot_count
            record = {
                "input": left.detach().cpu().clone(),
                "projected": out.detach().cpu().clone(),
                "norm_weight": norm.weight.detach().cpu().clone(),
                "variance_epsilon": norm.variance_epsilon,
                "weight": weight.detach().cpu().clone() if index == 0 else None,
                "input_stride": list(left.stride()),
                "weight_stride": list(right.stride()),
                "output_stride": list(out.stride()),
                "dp_rank": rank,
            }
            path = Path(os.environ["IRIS_OUTPUT_DIR"]) / f"latent-projection-{index}.pt"
            torch.save(record, path)
            worker.hero_latent_snapshot_count += 1
        return result

    extern_kernels.mm = observed_mm
    return {"dp_rank": rank, "capturing": True, "weight_shape": list(weight.shape), "maximum_captures": 2}


def install_attention_snapshot(worker) -> dict:
    """Capture the first local/global prefill at the native backend boundary."""
    runner = worker.model_runner
    assert runner.dp_rank == 0 and worker.vllm_config.parallel_config.data_parallel_size == 1
    assert not hasattr(worker, "hero_attention_snapshots")
    worker.hero_attention_snapshots = []
    for index in (0, 3):
        matches = [
            module for name, module in runner.model.named_modules() if name.endswith(f"layers.{index}.self_attn.attn")
        ]
        assert len(matches) == 1
        attention = matches[0]
        implementation = attention.impl
        assert type(implementation).__name__ == "FlashInferImpl"
        original = implementation.forward

        def capture(
            layer, query, key, value, kv_cache, attn_metadata, output, *args, _index=index, _call=original, **kwargs
        ):
            result = _call(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
            if query.shape[0] == 2048 and _index not in worker.hero_attention_snapshots:
                assert attn_metadata is not None and key is not None and value is not None
                tensors = {"query": query, "key": key, "value": value, "output": result}
                assert sum(t.numel() * t.element_size() for t in tensors.values()) < 80_000_000
                destination = Path(os.environ["IRIS_OUTPUT_DIR"]) / f"attention-layer-{_index}.pt"
                torch.save(
                    {
                        **{name: tensor.detach().cpu() for name, tensor in tensors.items()},
                        "cache_shape": list(kv_cache.shape),
                        "num_actual_tokens": attn_metadata.num_actual_tokens,
                        "layer": _index,
                        "scope": "First2048-row native prefill only; diagnostic synchronization, no timing.",
                    },
                    destination,
                )
                worker.hero_attention_snapshots.append(_index)
            return result

        implementation.forward = capture
    return {"dp_rank": runner.dp_rank, "layers": [0, 3], "prefill_rows": 2048}


class BatchTraceWorkerExtension:
    install_hero_batch_trace = install_batch_trace
    take_hero_batch_trace = take_batch_trace
    stop_hero_batch_trace = stop_batch_trace
    hero_cache_storage = cache_storage
    set_hero_shortconv_reference = set_shortconv_reference
    install_hero_moe_snapshot = install_moe_snapshot
    install_hero_latent_projection_snapshot = install_latent_projection_snapshot
    capture_hero_compiler_sources = capture_compiler_sources
    install_hero_attention_snapshot = install_attention_snapshot


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
