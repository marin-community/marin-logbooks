# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic local graph buckets with MoE communication outside every graph.

Install through the existing worker RPC only after capture. Native dispatch
first agrees whether every rank can replay a piecewise graph. A second CPU
collective exchanges the independently padded local sizes. Full graphs, DBO,
speculation, LoRA and other parallel layouts are outside this experiment.
"""

from dataclasses import replace

import torch
import torch.distributed as dist
from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import get_dp_group

MOE_SPLITTING_OPS = {"vllm::moe_forward", "vllm::moe_forward_shared"}


def validate_local_piecewise_config(worker) -> None:
    config = worker.vllm_config
    parallel = config.parallel_config
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
    assert MOE_SPLITTING_OPS <= set(config.compilation_config.splitting_ops)
    assert "vllm::grug_moe_short_conv" in config.compilation_config.splitting_ops
    assert parallel.tensor_parallel_size == parallel.pipeline_parallel_size == 1
    assert parallel.data_parallel_size > 1 and parallel.enable_expert_parallel
    assert parallel.all2all_backend == "allgather_reducescatter"
    assert not parallel.enable_dbo
    assert config.speculative_config is None and config.lora_config is None


def local_piecewise_dispatch(result, args, kwargs):
    descriptor, sync = result
    if descriptor.cg_mode != CUDAGraphMode.PIECEWISE or sync is None:
        return result
    assert descriptor.num_ubatches == 1 and not sync.eager
    assert kwargs.get("dp_sync") is None
    assert not kwargs.get("allow_ubatching", False)
    manager, num_reqs, num_tokens, uniform_token_count, dp_size, dp_rank = args
    local = manager.dispatch(
        num_reqs,
        num_tokens,
        uniform_token_count,
        max_query_len=kwargs.get("max_query_len"),
        num_active_loras=kwargs.get("num_active_loras", 0),
    )
    sizes = torch.zeros((2, dp_size), dtype=torch.int32, device="cpu")
    sizes[0, dp_rank] = local.num_tokens
    sizes[1, dp_rank] = local.cg_mode.value
    dist.all_reduce(sizes, group=get_dp_group().cpu_group)
    assert torch.all(sizes[1] == CUDAGraphMode.PIECEWISE.value)
    assert local.num_tokens >= num_tokens and local.num_ubatches == 1
    return local, replace(sync, num_tokens_across_dp=sizes[0])
