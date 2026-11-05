# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare a tiny local Hero engine with compiled uncaptured execution."""

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
import vllm
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode
from vllm.transformers_utils.configs.grugmoe import GrugMoeConfig


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


def worker_observation(worker) -> dict:
    # Read this process's counter after RPC deserialization in the worker.
    from vllm.compilation.counter import compilation_counter  # noqa: PLC0415

    config = worker.model_runner.compilation_config
    weights = hashlib.sha256()
    for name, parameter in sorted(worker.model_runner.model.named_parameters()):
        weights.update(name.encode())
        weights.update(parameter.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return {
        "mode": config.cudagraph_mode.name,
        "capture_sizes": list(config.cudagraph_capture_sizes),
        "splitting_ops": list(config.splitting_ops),
        "captures": compilation_counter.num_cudagraph_captured,
        "replays": worker._hero_validation_replays,
        "weights_sha256": weights.hexdigest(),
        "runner": type(worker.model_runner).__module__,
        "attention_groups": attention_observation(worker.model_runner),
    }


def install_replay_observer(worker) -> dict:
    # Resolve worker-owned graph state rather than serializing driver globals.
    from vllm.compilation.cuda_graph import CUDAGraphWrapper  # noqa: PLC0415
    from vllm.config import CUDAGraphMode  # noqa: PLC0415
    from vllm.forward_context import get_forward_context  # noqa: PLC0415

    worker._hero_validation_replays = 0
    original = CUDAGraphWrapper.__call__

    def observed(wrapper, *args, **kwargs):
        context = get_forward_context()
        entry = wrapper.concrete_cudagraph_entries.get(context.batch_descriptor)
        replay = (
            context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
            and wrapper.runtime_mode == CUDAGraphMode.PIECEWISE
            and entry is not None
            and entry.cudagraph is not None
        )
        result = original(wrapper, *args, **kwargs)
        # Exact-39e CUDAGraphWrapper reaches entry.cudagraph.replay() on this
        # branch. Count only successful calls, after the actual replay returns.
        if replay:
            worker._hero_validation_replays += 1
        return result

    CUDAGraphWrapper.__call__ = observed
    return worker_observation(worker)


def engine_kwargs(model_dir: Path, mode: str) -> dict:
    return dict(
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
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        kv_cache_memory_bytes=64 * 1024 * 1024,
        gpu_memory_utilization=0.05,
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode[mode],
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32] if mode == "PIECEWISE" else [],
            use_inductor_graph_partition=False,
        ),
        kernel_config={"moe_backend": "triton", "enable_flashinfer_autotune": False},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("NONE", "PIECEWISE"), required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-source", type=Path, required=True)
    args = parser.parse_args()

    # Import the installed wheel first, then reuse only the pinned test fixture.
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
    args.work.mkdir(parents=True)
    model_dir = args.work / "model"
    model_dir.mkdir()
    hf_config = GrugMoeConfig(
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
    )
    hf_config.save_pretrained(model_dir)
    llm = LLM(**engine_kwargs(model_dir, args.mode))
    try:
        before = llm.collective_rpc(install_replay_observer)[0]
        if before["runner"] != "vllm.v1.worker.gpu.model_runner" or before["mode"] != args.mode:
            raise ValueError(f"Engine runner/config differs from requested validation: {before}")
        if args.mode == "PIECEWISE" and (
            not before["capture_sizes"]
            or before["captures"] <= 0
            or "vllm::grug_moe_short_conv" not in before["splitting_ops"]
        ):
            raise ValueError("Engine did not capture the requested ShortConv-safe partitions")
        a, b, c = [1, 4, 7], [2, 5, 8, 11, 14], [3, 6, 9, 12, 15, 18, 21]
        observations = []
        for prompts in ([a, b, c], [c], [b, a]):
            parameters = [
                SamplingParams(temperature=0, max_tokens=4 - index, ignore_eos=True, logprobs=1, detokenize=False)
                for index in range(len(prompts))
            ]
            results = llm.generate([{"prompt_token_ids": prompt} for prompt in prompts], parameters, use_tqdm=False)
            batch = []
            for result, parameter in zip(results, parameters, strict=True):
                completion = result.outputs[0]
                if len(completion.token_ids) != parameter.max_tokens:
                    raise ValueError("Tiny engine did not finish the requested generation")
                chosen_logprobs = [
                    token_probs[token].logprob
                    for token, token_probs in zip(completion.token_ids, completion.logprobs, strict=True)
                ]
                if not torch.isfinite(torch.tensor(chosen_logprobs)).all():
                    raise ValueError("Tiny engine returned nonfinite token logprobs")
                batch.append({"tokens": list(completion.token_ids), "chosen_logprobs": chosen_logprobs})
            observations.append(batch)
        after = llm.collective_rpc(worker_observation)[0]
        if after["weights_sha256"] != before["weights_sha256"]:
            raise ValueError("Generation changed dummy model parameters")
        if args.mode == "PIECEWISE" and (after["replays"] <= 0 or after["captures"] != before["captures"]):
            raise ValueError("Tiny engine did not replay its captured partitions without recapture")
        args.output.write_text(
            json.dumps(
                {
                    "mode": args.mode,
                    "vllm_path": str(Path(vllm.__file__).resolve()),
                    "config": hf_config.to_dict(),
                    "before": before,
                    "after": after,
                    "weights_sha256": after["weights_sha256"],
                    "outputs": observations,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=30)


if __name__ == "__main__":
    main()
