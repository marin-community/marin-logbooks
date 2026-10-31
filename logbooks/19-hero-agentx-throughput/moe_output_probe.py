# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the buffered MoE operator against native Triton and plain Torch math.

Use one idle owned GPU and the exact private serving wheel. This component
diagnostic uses deterministic small weights, not the Hero checkpoint. It tests
the opaque operator's copy and numerical contract, not graph replay or speed.
"""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from vllm.forward_context import set_forward_context
from vllm.model_executor.models import grugmoe
from vllm.v1.worker.workspace import init_workspace_manager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-source-sha256", required=True)
    args = parser.parse_args()
    assert hashlib.sha256(Path(grugmoe.__file__).read_bytes()).hexdigest() == args.model_source_sha256
    assert torch.cuda.device_count() == 1
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    config = VllmConfig(kernel_config=KernelConfig(moe_backend="triton"))
    cfg = grugmoe.GrugMoeRuntimeConfig(
        vocab_size=64,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=0,
        num_shared_experts=0,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=4,
        num_kv_heads=1,
        head_dim=4,
        max_seq_len=128,
        sliding_window=8,
        initializer_std=0.02,
        disable_pko=True,
    ).validate()
    rows = []
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
                    parameter.copy_(
                        torch.linspace(
                            -0.2, 0.3, parameter.numel(), device=parameter.device, dtype=parameter.dtype
                        ).reshape_as(parameter)
                    )
                experts = module.experts.routed_experts
                experts.quant_method.process_weights_after_loading(experts)
                init_workspace_manager(torch.device("cuda"))
                for count in [1, 3, 8, 17, 32, 64, 2, 1]:
                    hidden = torch.linspace(-0.6, 0.7, count * cfg.hidden_dim, device="cuda").to(torch.bfloat16)
                    hidden = hidden.reshape(count, cfg.hidden_dim)
                    logits = F.linear(hidden.float(), module.router.weight)
                    with set_forward_context(None, config, num_tokens=count):
                        native = module.experts(hidden.clone(), logits.clone())
                        output = torch.full_like(hidden, torch.nan)
                        torch.ops.vllm.grug_moe_experts_out(
                            hidden.clone(), logits.clone(), output, module.experts.layer_name
                        )
                        reference = module._forward_torch_reference(hidden.float(), hidden.float())
                    assert torch.isfinite(output).all()
                    torch.testing.assert_close(output, native, atol=0, rtol=0)
                    # Reuse the existing BF16 Grug layer test's tolerances.
                    torch.testing.assert_close(output.float(), reference, atol=2e-2, rtol=2e-2)
                    delta = (output.float() - reference).abs()
                    rows.append(
                        {
                            "token_rows": count,
                            "native_max_absolute_delta": (output - native).abs().max().item(),
                            "torch_fp32_max_absolute_delta": delta.max().item(),
                            "torch_fp32_mean_absolute_delta": delta.mean().item(),
                        }
                    )
        finally:
            cleanup_dist_env_and_memory()
    args.output.parent.mkdir(exist_ok=False)
    result = {
        "status": "native_exact_and_existing_torch_tolerances_passed",
        "attempt_uid": os.environ["IRIS_ATTEMPT_UID"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_source_sha256": args.model_source_sha256,
        "torch_reference_atol": 2e-2,
        "torch_reference_rtol": 2e-2,
        "rows": rows,
        "scope": "One-GPU small deterministic BF16 Triton component check, no graph or full-Hero qualification.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
