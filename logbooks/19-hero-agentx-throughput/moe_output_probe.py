# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the buffered MoE operator against a native backend and plain Torch math.

Use one idle owned GPU and the exact private serving wheel. This component
diagnostic uses deterministic synthetic weights, not the Hero checkpoint. It tests
the opaque operator's copy and numerical contract, not graph replay or speed.
"""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from vllm.forward_context import set_forward_context
from vllm.model_executor.models import grugmoe
from vllm.v1.worker.workspace import init_workspace_manager

ROW_COUNTS = (1, 3, 8, 17, 32, 64, 2, 1)


def probe_config(
    hidden_dim: int = 16, intermediate_dim: int = 16, num_experts: int = 4, top_k: int = 2
) -> grugmoe.GrugMoeRuntimeConfig:
    return grugmoe.GrugMoeRuntimeConfig(
        vocab_size=64,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        shared_expert_intermediate_dim=0,
        num_experts=num_experts,
        num_experts_per_token=top_k,
        num_layers=1,
        num_heads=4,
        num_kv_heads=1,
        head_dim=hidden_dim // 4,
        max_seq_len=128,
        sliding_window=8,
        initializer_std=0.02,
        disable_pko=True,
    ).validate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-source-sha256", required=True)
    parser.add_argument("--backend", choices=("triton", "flashinfer_cutlass"), default="triton")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--intermediate-dim", type=int, default=16)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--row-counts", type=int, nargs="+", default=ROW_COUNTS)
    parser.add_argument("--normal-weights", action="store_true")
    parser.add_argument("--normal-inputs", action="store_true")
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--reference-chunk-rows", type=int, default=4)
    args = parser.parse_args()
    assert args.reference_chunk_rows > 0
    assert hashlib.sha256(Path(grugmoe.__file__).read_bytes()).hexdigest() == args.model_source_sha256
    cfg = probe_config(args.hidden_dim, args.intermediate_dim, args.num_experts, args.top_k)
    assert torch.cuda.device_count() == 1
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    config = VllmConfig(kernel_config=KernelConfig(moe_backend=args.backend))
    rows = []
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as temporary, set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="file://" + str(Path(temporary) / "gloo"),
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        try:
            module = grugmoe.GrugMoeMLP(cfg, params_dtype=torch.bfloat16, prefix="probe.mlp").cuda()
            with torch.no_grad():
                for parameter in module.parameters():
                    if args.normal_weights:
                        parameter.copy_(torch.randn_like(parameter) * 0.02)
                    else:
                        parameter.copy_(
                            torch.linspace(
                                -0.2, 0.3, parameter.numel(), device=parameter.device, dtype=parameter.dtype
                            ).reshape_as(parameter)
                        )
                experts = module.experts.routed_experts
                inputs = {}
                for count in args.row_counts:
                    inputs[count] = (
                        torch.randn(count, cfg.hidden_dim, device="cuda", dtype=torch.bfloat16)
                        if args.normal_inputs
                        else torch.linspace(-0.6, 0.7, count * cfg.hidden_dim, device="cuda")
                        .to(torch.bfloat16)
                        .reshape(count, cfg.hidden_dim)
                    )
                # Backend conversion may transpose the weights. Use the canonical
                # weights for the independent reference before that conversion.
                references = {}
                for count, hidden in inputs.items():
                    with set_forward_context(None, config, num_tokens=count):
                        # The plain reference gathers three expert matrices per
                        # token. Bound that scratch space while keeping each
                        # token's routing and FP32 arithmetic unchanged.
                        references[count] = torch.cat(
                            [
                                module._forward_torch_reference(chunk.float(), chunk.float())
                                for chunk in hidden.split(args.reference_chunk_rows)
                            ]
                        )
                experts.quant_method.process_weights_after_loading(experts)
                init_workspace_manager(torch.device("cuda"))
                backend = str(experts.quant_method.unquantized_backend)
                experts_class = experts.quant_method.experts_cls.__name__
                for count in args.row_counts:
                    hidden = inputs[count]
                    logits = F.linear(hidden.float(), module.router.weight)
                    with set_forward_context(None, config, num_tokens=count):
                        native = module.experts(hidden.clone(), logits.clone())
                        if args.native_only:
                            output = native
                        else:
                            output = torch.full_like(hidden, torch.nan)
                            torch.ops.vllm.grug_moe_experts_out(
                                hidden.clone(), logits.clone(), output, module.experts.layer_name
                            )
                            torch.testing.assert_close(output, native, atol=0, rtol=0)
                        reference = references[count]
                    assert torch.isfinite(output).all()
                    # Reuse the existing BF16 Grug layer test's tolerances.
                    torch.testing.assert_close(output.float(), reference, atol=2e-2, rtol=2e-2)
                    delta = (output.float() - reference).abs()
                    rows.append(
                        {
                            "token_rows": count,
                            "native_max_absolute_delta": (
                                None if args.native_only else (output - native).abs().max().item()
                            ),
                            "torch_fp32_max_absolute_delta": delta.max().item(),
                            "torch_fp32_mean_absolute_delta": delta.mean().item(),
                        }
                    )
        finally:
            cleanup_dist_env_and_memory()
    args.output.parent.mkdir(exist_ok=False)
    result = {
        "status": (
            "native_backend_existing_torch_tolerances_passed"
            if args.native_only
            else "native_exact_and_existing_torch_tolerances_passed"
        ),
        "attempt_uid": os.environ["IRIS_ATTEMPT_UID"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_source_sha256": args.model_source_sha256,
        "torch_reference_atol": 2e-2,
        "torch_reference_rtol": 2e-2,
        "requested_backend": args.backend,
        "actual_backend": backend,
        "experts_class": experts_class,
        "output_buffer_contract_checked": not args.native_only,
        "weight_pattern": "seed42_normal_std0.02" if args.normal_weights else "linspace_minus0.2_to0.3",
        "input_pattern": "seed42_normal" if args.normal_inputs else "linspace_minus0.6_to0.7",
        "fp32_reference_matmul": "highest; TF32 disabled",
        "reference_chunk_rows": args.reference_chunk_rows,
        "configuration": asdict(cfg),
        "rows": rows,
        "scope": f"One-GPU synthetic BF16 {args.backend} component check, no graph or full-Hero qualification.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
