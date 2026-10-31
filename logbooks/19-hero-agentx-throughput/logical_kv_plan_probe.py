# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Use the pinned native CPU planner to assess six versus twelve global KV heads.

Reconstruct the actual retained per-layer cache specs and tensor-storage budget.
This is a hypothetical cache-layout plan, without model edits or GPU execution.
"""

import argparse
import ast
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    get_max_concurrency_for_kv_cache_config,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec


def integer(spec: str, field: str) -> int:
    match = re.search(rf"\b{field}=(-?\d+)", spec)
    assert match is not None, (field, spec)
    return int(match[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.config.read_bytes()).hexdigest() == (
        "4cf93824957296e3b36c458ab4ae598bb7616f202a74687a3a8ed64dac9a9ebf"
    )
    retained = json.loads(args.storage.read_text())["results"]
    assert len(retained) == 1
    storage = retained[0]
    specs = {}
    for group in storage["cache_groups"]:
        text = group["spec"]
        if group["spec_type"] == "MambaSpec":
            shapes = ast.literal_eval(re.search(r"shapes=(.*?), dtypes=", text)[1])
            spec = MambaSpec(
                block_size=integer(text, "block_size"),
                shapes=shapes,
                dtypes=(torch.bfloat16,),
                page_size_padded=integer(text, "page_size_padded"),
                mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
                mamba_cache_mode="align",
                num_speculative_blocks=0,
                num_prefill_checkpoint_blocks=0,
            )
        else:
            kwargs = {
                "block_size": integer(text, "block_size"),
                "num_kv_heads": integer(text, "num_kv_heads"),
                "head_size": integer(text, "head_size"),
                "dtype": torch.bfloat16,
            }
            if group["spec_type"] == "SlidingWindowSpec":
                spec = SlidingWindowSpec(**kwargs, sliding_window=integer(text, "sliding_window"))
            else:
                assert group["spec_type"] == "FullAttentionSpec"
                spec = FullAttentionSpec(**kwargs)
        for layer in range(group["layers"]):
            specs[f"retained_group_{group['group']}_layer_{layer}"] = spec
    model = ModelConfig(
        model=str(args.config.parent),
        tokenizer=str(args.config.parent),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=262144,
    )
    config = VllmConfig(
        model_config=model,
        cache_config=CacheConfig(
            block_size=16, enable_prefix_caching=True, mamba_cache_mode="align", mamba_block_size=16
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=262144, max_num_batched_tokens=2048, max_num_seqs=512, async_scheduling=True
        ),
    )
    records = {}
    for compact in (False, True):
        candidate = {
            name: replace(spec, num_kv_heads=6) if compact and type(spec) is FullAttentionSpec else spec
            for name, spec in specs.items()
        }
        groups = get_kv_cache_groups(config, candidate)
        plan = get_kv_cache_config_from_groups(config, groups, storage["unique_cuda_storage_bytes"])
        concurrency = get_max_concurrency_for_kv_cache_config(config, plan)
        assert plan.num_blocks == storage["native_num_blocks"]
        assert len(groups) == len(storage["cache_groups"]) == 16
        records["logical_global_6" if compact else "stored_global_12"] = {
            "num_blocks": plan.num_blocks,
            "physical_cache_bytes": sum(tensor.size for tensor in plan.kv_cache_tensors),
            "max_context_concurrency": concurrency,
            "group_aware_token_capacity": concurrency * 262144,
            "groups": [
                {
                    "type": type(group.kv_cache_spec).__name__,
                    "layers": len(group.layer_names),
                    "block_size": group.kv_cache_spec.block_size,
                    "page_size_bytes": group.kv_cache_spec.page_size_bytes,
                    "num_kv_heads": getattr(group.kv_cache_spec, "num_kv_heads", None),
                }
                for group in groups
            ],
        }
    result = {
        "status": "hypothetical_native_cpu_cache_plans_complete",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "storage_input_sha256": hashlib.sha256(args.storage.read_bytes()).hexdigest(),
        "max_in_flight_tokens": config.max_in_flight_tokens,
        "plans": records,
        "capacity_ratio": (
            records["logical_global_6"]["group_aware_token_capacity"]
            / records["stored_global_12"]["group_aware_token_capacity"]
        ),
        "scope": (
            "Native CPU grouping/allocation/capacity functions on retained specs and budget; "
            "no edited model or GPU result."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
