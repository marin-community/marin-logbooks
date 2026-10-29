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
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("NONE", "PIECEWISE"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compilation = {
        "mode": 3,
        "cudagraph_mode": args.mode,
        "cudagraph_capture_sizes": [1, 2, 4, 8],
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
        "compilation_requested": compilation,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in ("vllm", "torch", "flashinfer-python")},
        "rows": [],
    }
    started = time.monotonic()
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=128,
        max_num_seqs=8,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.15,
        enforce_eager=False,
        enable_prefix_caching=True,
        kernel_config={"moe_backend": "triton"},
        compilation_config=compilation,
        seed=42,
    )
    try:
        actual = llm.llm_engine.vllm_config
        proof.update(
            startup_seconds=time.monotonic() - started,
            actual_cudagraph_mode=actual.compilation_config.cudagraph_mode.name,
            use_v2_model_runner=actual.use_v2_model_runner,
        )
        assert proof["actual_cudagraph_mode"] == args.mode
        prefix = [1, 2, 3] * 16
        prompts = [prefix + suffix for suffix in ([1, 2, 3], [3, 2], [4, 5, 6])]
        for branches in ([0], [1, 0], [2, 0, 1], [0]) * 2:
            started = time.monotonic()
            outputs = llm.generate(
                [{"prompt_token_ids": prompts[index]} for index in branches],
                SamplingParams(temperature=0, seed=42, max_tokens=8, min_tokens=8, ignore_eos=True, logprobs=5),
                use_tqdm=False,
            )
            row = {
                "branches": branches,
                "seconds": time.monotonic() - started,
                "tokens": [list(output.outputs[0].token_ids) for output in outputs],
                "logprobs": [
                    [{str(key): value.logprob for key, value in item.items()} for item in output.outputs[0].logprobs]
                    for output in outputs
                ],
            }
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
