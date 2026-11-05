# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify routed-expert capture on one GPU and a local DP4/EP4 engine."""

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch
import vllm
from hero_shortconv_route_contract import request_waves, route_record
from vllm import SamplingParams
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.transformers_utils.configs.grugmoe import GrugMoeConfig
from vllm.v1.engine.async_llm import AsyncLLM


def attention_observation(runner) -> list[dict]:
    from vllm.v1.kv_cache_interface import MambaSpec  # noqa: PLC0415

    groups = [
        {
            "backend": group.backend.get_name(),
            "cache_spec": type(group.kv_cache_spec).__name__,
            "kind": "state" if isinstance(group.kv_cache_spec, MambaSpec) else "attention",
            "layers": list(group.layer_names),
        }
        for cache_group in runner.attn_groups
        for group in cache_group
    ]
    standard = [group for group in groups if group["kind"] == "attention"]
    if not standard or any(group["backend"] != "FLASH_ATTN" for group in standard):
        raise ValueError(f"Resolved standard-attention backend differs from Hero FLASH_ATTN: {groups}")
    return groups


def snapshot_worker(worker, directory: str, phase: str) -> dict:
    # Resolve process-owned state inside the worker RPC, not driver globals.
    from vllm.compilation.counter import compilation_counter  # noqa: PLC0415
    from vllm.distributed import get_ep_group, get_pp_group, get_tp_group  # noqa: PLC0415

    runner = worker.model_runner
    config = runner.compilation_config
    parallel = runner.vllm_config.parallel_config
    digest = hashlib.sha256()
    for name, parameter in sorted(runner.model.named_parameters()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    result = {
        "rank": parallel.data_parallel_rank,
        "dp": parallel.data_parallel_size,
        "tp": get_tp_group().world_size,
        "pp": get_pp_group().world_size,
        "ep": get_ep_group().world_size,
        "mode": config.cudagraph_mode.name,
        "capture_sizes": list(config.cudagraph_capture_sizes),
        "splitting_ops": list(config.splitting_ops),
        "captures": compilation_counter.num_cudagraph_captured,
        "replays": worker._hero_route_validation_replays,
        "dummy_batches": worker._hero_route_validation_dummy_batches,
        "route_token_counts": sorted(worker._hero_route_validation_token_counts),
        "route_buffer_shape": list(runner.routed_experts_capturer.device_buffer.shape),
        "weights_sha256": digest.hexdigest(),
        "runner": type(runner).__module__,
        "attention_groups": attention_observation(runner),
    }
    Path(directory, f"{phase}-dp{parallel.data_parallel_rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def install_observers(worker, directory: str) -> dict:
    from vllm.compilation.cuda_graph import CUDAGraphWrapper  # noqa: PLC0415
    from vllm.config import CUDAGraphMode  # noqa: PLC0415
    from vllm.forward_context import get_forward_context  # noqa: PLC0415

    if worker.model_runner.routed_experts_capturer is None:
        raise ValueError("The requested routed-expert capturer was not initialized")
    worker._hero_route_validation_replays = 0
    worker._hero_route_validation_dummy_batches = 0
    worker._hero_route_validation_token_counts = set()
    original_graph = CUDAGraphWrapper.__call__
    original_dummy = worker.execute_dummy_batch
    capturer = worker.model_runner.routed_experts_capturer
    original_routes = capturer.get_routed_experts

    def observed_graph(wrapper, *args, **kwargs):
        context = get_forward_context()
        entry = wrapper.concrete_cudagraph_entries.get(context.batch_descriptor)
        replay = (
            context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
            and wrapper.runtime_mode == CUDAGraphMode.PIECEWISE
            and entry is not None
            and entry.cudagraph is not None
        )
        result = original_graph(wrapper, *args, **kwargs)
        if replay:
            worker._hero_route_validation_replays += 1
        return result

    def observed_dummy(*args, **kwargs):
        result = original_dummy(*args, **kwargs)
        worker._hero_route_validation_dummy_batches += 1
        return result

    def observed_routes(slot_mappings, num_tokens):
        result = original_routes(slot_mappings, num_tokens)
        worker._hero_route_validation_token_counts.add(int(num_tokens))
        return result

    CUDAGraphWrapper.__call__ = observed_graph
    worker.execute_dummy_batch = observed_dummy
    capturer.get_routed_experts = observed_routes
    return snapshot_worker(worker, directory, "before")


def read_snapshots(directory: Path, phase: str, dp_size: int) -> list[dict]:
    snapshots = [json.loads(path.read_text()) for path in directory.glob(f"{phase}-dp*.json")]
    if {item["rank"] for item in snapshots} != set(range(dp_size)) or len(snapshots) != dp_size:
        raise ValueError(f"Worker receipts do not cover all {dp_size} DP ranks at {phase}")
    return sorted(snapshots, key=lambda item: item["rank"])


def write_model(args) -> Path:
    # vLLM was imported from the installed wheel before this test-only path.
    sys.path.insert(0, str(args.test_source))
    from tests.models.test_grugmoe import _tiny_hero_config  # noqa: PLC0415

    cfg = replace(
        _tiny_hero_config(),
        hidden_dim=128,
        intermediate_dim=64,
        shared_expert_intermediate_dim=64,
        latent_dim=64,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        local_kv_heads=1,
        global_kv_heads=1,
        head_dim=64,
        max_seq_len=128,
        sliding_window=64,
        global_every=2,
    ).validate()
    model_dir = args.work / "model"
    model_dir.mkdir()
    GrugMoeConfig(
        architectures=["GrugMoeForCausalLM"],
        grugmoe_artifact_schema_version=2,
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_dim,
        moe_intermediate_size=cfg.intermediate_dim,
        shared_expert_intermediate_size=cfg.shared_expert_intermediate_dim,
        num_shared_experts=cfg.num_shared_experts,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_token,
        latent_dim=cfg.latent_dim,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_kv_heads,
        local_kv_heads=cfg.local_kv_heads,
        global_kv_heads=cfg.global_kv_heads,
        head_dim=cfg.head_dim,
        max_position_embeddings=cfg.max_seq_len,
        sliding_window=cfg.sliding_window,
        global_every=cfg.global_every,
        rope_fused=cfg.rope_fused,
        sconv=True,
        sconv_kernel=cfg.sconv_kernel,
        sconv_sites=list(cfg.sconv_sites),
        disable_pko=cfg.disable_pko,
        disable_long_rope=cfg.disable_long_rope,
    ).save_pretrained(model_dir)
    return model_dir


def engine_arguments(model_dir: Path, mode: str, dp_size: int) -> AsyncEngineArgs:
    return AsyncEngineArgs(
        model=str(model_dir),
        skip_tokenizer_init=True,
        load_format="dummy",
        dtype="bfloat16",
        enforce_eager=False,
        attention_backend="FLASH_ATTN",
        seed=42,
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=dp_size,
        data_parallel_size_local=dp_size,
        data_parallel_backend="mp",
        distributed_executor_backend="mp",
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
        enable_return_routed_experts=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        kv_cache_memory_bytes=64 * 1024 * 1024,
        gpu_memory_utilization=0.05,
        logprobs_mode="processed_logprobs",
        generation_config="vllm",
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode[mode],
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32] if mode == "PIECEWISE" else [],
            use_inductor_graph_partition=False,
        ),
        kernel_config={"moe_backend": "triton", "enable_flashinfer_autotune": False},
    )


async def generate_case(engine: AsyncLLM, case, dp_size: int) -> dict:
    sampling = SamplingParams(
        temperature=0,
        max_tokens=case.output_tokens,
        ignore_eos=True,
        logprobs=1,
        detokenize=False,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    final = None
    async for result in engine.generate(
        prompt={"prompt_token_ids": list(case.prompt)},
        sampling_params=sampling,
        request_id=case.request_id,
        data_parallel_rank=case.rank if dp_size > 1 else None,
    ):
        final = result
    if final is None or not final.finished or final.request_id != case.request_id:
        raise ValueError(f"Request did not produce its final output: {case.request_id}")
    if list(final.prompt_token_ids) != list(case.prompt):
        raise ValueError("Engine changed the supplied token prompt")
    completion = final.outputs[0]
    if len(completion.token_ids) != case.output_tokens:
        raise ValueError("Engine did not produce the exact requested output length")
    logprobs = [values[token].logprob for token, values in zip(completion.token_ids, completion.logprobs, strict=True)]
    if not torch.isfinite(torch.tensor(logprobs)).all():
        raise ValueError("Engine returned nonfinite chosen-token logprobs")
    return {
        "request": asdict(case),
        "tokens": list(completion.token_ids),
        "chosen_logprobs": logprobs,
        **route_record(completion.routed_experts, len(final.prompt_token_ids), len(completion.token_ids)),
    }


async def run(args) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.dp_size:
        raise ValueError("This qualification requires the requested visible CUDA devices")
    args.work.mkdir(parents=True)
    receipts = args.work / "worker-receipts"
    receipts.mkdir()
    model_dir = write_model(args)
    engine = AsyncLLM.from_engine_args(engine_arguments(model_dir, args.mode, args.dp_size))
    try:
        # Pinned DPLBAsyncMPClient fans RPC out to every engine but returns only
        # the first engine's result; rank-specific files preserve all results.
        await engine.collective_rpc(install_observers, timeout=120, args=(str(receipts),))
        before = read_snapshots(receipts, "before", args.dp_size)
        for item in before:
            if (item["tp"], item["pp"], item["dp"], item["ep"]) != (1, 1, args.dp_size, args.dp_size):
                raise ValueError("Worker process groups do not match TP1/PP1/DP=EP geometry")
            if item["runner"] != "vllm.v1.worker.gpu.model_runner" or item["mode"] != args.mode:
                raise ValueError("Worker is using a different runner or graph mode")
            if item["route_buffer_shape"][1:] != [2, 2]:
                raise ValueError("Worker route buffer geometry differs from the tiny Hero fixture")
            if args.mode == "PIECEWISE" and (
                item["captures"] <= 0
                or not item["capture_sizes"]
                or "vllm::grug_moe_short_conv" not in item["splitting_ops"]
            ):
                raise ValueError("A DP rank did not capture ShortConv-safe partitions")
        outputs = []
        phases = []
        previous = before
        for index, wave in enumerate(request_waves(args.dp_size)):
            # Keep request order and drain sibling tasks if any request fails.
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(generate_case(engine, case, args.dp_size)) for case in wave]
            outputs.append([task.result() for task in tasks])
            phase = f"after-{index}"
            await engine.collective_rpc(snapshot_worker, timeout=120, args=(str(receipts), phase))
            current = read_snapshots(receipts, phase, args.dp_size)
            if index == 1 and args.dp_size > 1:
                idle = set(range(args.dp_size)) - {case.rank for case in wave}
                if any(current[rank]["dummy_batches"] <= previous[rank]["dummy_batches"] for rank in idle):
                    raise ValueError("The idle-peer wave did not execute the real DP dummy branch on every idle rank")
            phases.append(current)
            previous = current
        after = phases[-1]
        for initial, final in zip(before, after, strict=True):
            if initial["weights_sha256"] != final["weights_sha256"]:
                raise ValueError("Generation changed the dummy model weights")
            if args.mode == "PIECEWISE" and (final["replays"] <= 0 or final["captures"] != initial["captures"]):
                raise ValueError("A DP rank failed to replay captured partitions without recapture")
        args.output.write_text(
            json.dumps(
                {
                    "mode": args.mode,
                    "dp_size": args.dp_size,
                    "vllm_path": str(Path(vllm.__file__).resolve()),
                    "weights_sha256": [item["weights_sha256"] for item in after],
                    "before": before,
                    "phases": phases,
                    "outputs": outputs,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        engine.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("NONE", "PIECEWISE"), required=True)
    parser.add_argument("--dp-size", type=int, choices=(1, 4), required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-source", type=Path, required=True)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    if args.check_config:
        args.work.mkdir(parents=True)
        model_dir = write_model(args)
        config = engine_arguments(model_dir, args.mode, args.dp_size)
        args.output.write_text(
            json.dumps(
                {
                    "mode": config.compilation_config.cudagraph_mode.name,
                    "compilation": config.compilation_config.mode.name,
                    "tensor_parallel_size": config.tensor_parallel_size,
                    "pipeline_parallel_size": config.pipeline_parallel_size,
                    "data_parallel_size": config.data_parallel_size,
                    "data_parallel_size_local": config.data_parallel_size_local,
                    "enable_expert_parallel": config.enable_expert_parallel,
                    "enable_return_routed_experts": config.enable_return_routed_experts,
                    "data_parallel_backend": config.data_parallel_backend,
                    "enforce_eager": config.enforce_eager,
                    "attention_backend": config.attention_backend,
                    "waves": [[asdict(case) for case in wave] for wave in request_waves(args.dp_size)],
                    "model_config": json.loads((model_dir / "config.json").read_text()),
                    "scope": (
                        "CPU construction of actual pinned AsyncEngineArgs and local Hero config; no engine instantiated"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        return
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
