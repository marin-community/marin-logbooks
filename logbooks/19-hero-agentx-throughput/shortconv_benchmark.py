# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Time the native ShortConv decode path with original and direct output buffers.

Use one otherwise idle GPU inside the held allocation. This measures complete
module calls, including Python dispatch and allocation; it predicts no AgentX
speedup. The independent numerical regression is a separate prerequisite.
"""

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import torch
from native_shortconv_probe import install_forward
from vllm.config import VllmConfig, get_current_vllm_config, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.grugmoe import GrugMoeShortConv
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata


def benchmark(config: dict, candidate_source: Path) -> list[dict]:
    original = GrugMoeShortConv.forward_impl
    install_forward(candidate_source, config["method_file_sha256"])
    candidate = GrugMoeShortConv.forward_impl
    results = []
    torch.manual_seed(42)
    for dim in (1536, 6144):
        for batch in (1, 2, 8, 32, 64):
            prefix = f"hero.benchmark.{dim}.{batch}"
            module = GrugMoeShortConv(dim, 4, torch.bfloat16, None, prefix).cuda()
            module.weight.data.normal_(0, 0.1)
            module.kv_cache = (torch.zeros(batch + 1, 3, dim, dtype=torch.bfloat16, device="cuda"),)
            hidden = torch.randn(batch, dim, dtype=torch.bfloat16, device="cuda")
            metadata = ShortConvAttentionMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=batch,
                num_decode_tokens=batch,
                num_reqs=batch,
                query_start_loc_p=None,
                has_initial_states_p=None,
                state_indices_tensor_p=torch.empty(0, device="cuda", dtype=torch.int32),
                state_indices_tensor_d=torch.arange(1, batch + 1, device="cuda", dtype=torch.int32),
                num_accepted_tokens=None,
                query_start_loc_d=torch.arange(batch + 1, device="cuda", dtype=torch.int32),
                block_idx_last_scheduled_token=None,
                block_idx_first_scheduled_token_p=None,
                block_idx_last_computed_token=None,
                block_idx_last_scheduled_token_prev_step=None,
                num_computed_tokens_p=None,
                seq_lens=torch.full((batch,), 4, device="cuda"),
            )
            with (
                torch.inference_mode(),
                set_forward_context({prefix: metadata}, get_current_vllm_config(), num_tokens=batch),
            ):
                # Alternate implementations to reduce order/clock bias. All JIT
                # work and the warmup calls are outside the measured loop.
                for repetition in range(3):
                    for name, implementation in (("baseline", original), ("candidate", candidate)):
                        GrugMoeShortConv.forward_impl = implementation
                        for _ in range(10):
                            module(hidden)
                        torch.cuda.synchronize()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        submitted = time.perf_counter()
                        for _ in range(100):
                            module(hidden)
                        submit_seconds = time.perf_counter() - submitted
                        end.record()
                        end.synchronize()
                        results.append(
                            {
                                "implementation": name,
                                "dim": dim,
                                "batch": batch,
                                "repetition": repetition,
                                "calls": 100,
                                "host_submit_microseconds_per_call": submit_seconds * 1e4,
                                "cuda_elapsed_microseconds_per_call": start.elapsed_time(end) * 10,
                            }
                        )
    GrugMoeShortConv.forward_impl = original
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())
    with tempfile.TemporaryDirectory() as directory, set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1, rank=0, distributed_init_method=f"file://{directory}/rendezvous", local_rank=0, backend="gloo"
        )
        initialize_model_parallel(1, 1)
        try:
            results = benchmark(config, args.candidate_source)
        finally:
            cleanup_dist_env_and_memory()
    record = {
        "manifest": config,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(0),
        "results": results,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
