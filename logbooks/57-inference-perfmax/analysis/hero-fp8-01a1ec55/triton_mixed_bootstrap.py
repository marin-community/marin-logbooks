# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check native ignored BF16 and online FP8 MoE layers in one installed wheel."""

import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
import vllm
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.config.quantization import QuantizationConfigArgs
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import grugmoe
from vllm.model_executor.models.grugmoe import GrugMoeMLP, _try_load_grug_expert_weight
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager


def maximum(left, right):
    return float((left.float() - right.float()).abs().max())


def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    output = Path(payload["output"])
    root = Path(payload["source_root"]).resolve()
    assert not Path(vllm.__file__).resolve().is_relative_to(root)
    assert hashlib.sha256(Path(grugmoe.__file__).read_bytes()).hexdigest() == payload["model_sha256"]
    assert importlib.metadata.version("vllm") == payload["candidate_manifest"]["version"]
    sys.path.insert(0, str(root))
    # These are the pinned source's independent BF16 reference and tiny Hero
    # configuration. The imported vllm package remains the installed wheel.
    from tests.models.test_grugmoe import _reference_moe, _tiny_hero_config  # noqa: PLC0415

    record = {
        "status": "running",
        "scope": "Native mixed-precision loader/output and CUDA-graph component, not full-model qualification",
        "source_commit": payload["source_commit"],
        "wheel_sha256": payload["candidate_manifest"]["wheels"][0]["sha256"],
        "model_sha256": payload["model_sha256"],
        "installed_vllm": vllm.__file__,
        "activation_precision": "Native Triton dynamic per-tensor FP8 for quantized layers; BF16 for ignored layers",
        "kv_precision": "Not exercised by this MoE component; full-model gates must separately verify BF16 KV",
        "independent_reference_limits": {"atol": 0.05, "rtol": 0.05},
        "started_unix": time.time(),
        "layers": [],
        "cases": [],
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
    config = VllmConfig(kernel_config=KernelConfig(moe_backend="triton"))
    config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    cfg = replace(
        _tiny_hero_config(),
        hidden_dim=128,
        latent_dim=64,
        intermediate_dim=64,
        shared_expert_intermediate_dim=64,
        num_shared_experts=2,
        head_dim=32,
    ).validate()
    quantization = OnlineQuantizationConfig(
        QuantizationConfigArgs(moe="fp8_per_tensor", ignore=["model.layers.0.mlp.experts*"])
    )
    with tempfile.NamedTemporaryFile(delete=False) as temporary:
        temporary_name = temporary.name
    try:
        with set_current_vllm_config(config), torch.inference_mode():
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temporary_name}",
                local_rank=0,
                backend="gloo",
            )
            initialize_model_parallel(1, 1)
            modules, references = [], []
            with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
                for index in range(2):
                    references.append(GrugMoeMLP(cfg, prefix=f"reference.layers.{index}.mlp"))
                    modules.append(GrugMoeMLP(cfg, quant_config=quantization, prefix=f"model.layers.{index}.mlp"))
            for index, (reference, module) in enumerate(zip(references, modules, strict=True)):
                torch.manual_seed(42 + index)
                for name, weight in reference.named_parameters():
                    if name == "latent_norm.weight":
                        weight.fill_(1)
                    elif name == "router.bias":
                        weight.copy_(torch.linspace(0.2, -0.2, weight.numel(), device="cuda"))
                    else:
                        weight.copy_(torch.randn_like(weight.float()).mul_(0.04))
                reference.router.weight.copy_(reference.router.weight.bfloat16().float())
                destination = dict(module.named_parameters())
                for name, weight in reference.named_parameters():
                    if name == "experts.routed_experts.w13_weight":
                        for projection, shard in (
                            ("gate_proj", weight[:, : cfg.intermediate_dim]),
                            ("up_proj", weight[:, cfg.intermediate_dim :]),
                        ):
                            assert _try_load_grug_expert_weight(f"experts.{projection}.weight", shard, destination)
                    elif name == "experts.routed_experts.w2_weight":
                        assert _try_load_grug_expert_weight("experts.down_proj.weight", weight, destination)
                    else:
                        parameter = destination[name]
                        getattr(parameter, "weight_loader", default_weight_loader)(parameter, weight)
                reference.experts.routed_experts.quant_method.process_weights_after_loading(
                    reference.experts.routed_experts
                )
                experts = module.experts.routed_experts
                experts.quant_method.process_weights_after_loading(experts)
                method = experts.quant_method
                quant = method.moe_quant_config
                item = {
                    "layer": index,
                    "method": type(method).__name__,
                    "backend": str(getattr(method, "fp8_backend", getattr(method, "unquantized_backend", None))),
                    "weight_dtypes": [str(experts.w13_weight.dtype), str(experts.w2_weight.dtype)],
                    "weight_bytes": sum(w.numel() * w.element_size() for w in (experts.w13_weight, experts.w2_weight)),
                    "scale_shapes": {
                        name: list(getattr(experts, name).shape)
                        for name in ("w13_weight_scale", "w2_weight_scale")
                        if hasattr(experts, name)
                    },
                    "consumer_activation_dtype": str(getattr(quant, "quant_dtype", None)),
                    "consumer_weight_dtype": str(getattr(quant, "weight_quant_dtype", None)),
                    "router_dtype": str(module.router.weight.dtype),
                    "projection_dtypes": [
                        str(module.latent_down_proj.weight.dtype),
                        str(module.latent_up_proj.weight.dtype),
                    ],
                }
                expected = "torch.bfloat16" if index == 0 else "torch.float8_e4m3fn"
                assert item["weight_dtypes"] == [expected, expected], item
                assert item["method"] == ("UnquantizedFusedMoEMethod" if index == 0 else "Fp8PerTensorOnlineMoEMethod")
                assert item["backend"].endswith("TRITON"), item
                assert item["consumer_activation_dtype"] == ("None" if index == 0 else "torch.float8_e4m3fn"), item
                assert item["router_dtype"] == "torch.float32", item
                assert item["projection_dtypes"] == ["torch.bfloat16", "torch.bfloat16"], item
                record["layers"].append(item)
            init_workspace_manager(torch.device("cuda"))
            for count in (1, 7, 128, 129):
                inputs = torch.randn(count, cfg.hidden_dim, device="cuda", dtype=torch.bfloat16) / 4
                for index, (reference, module) in enumerate(zip(references, modules, strict=True)):
                    expected = _reference_moe(inputs, reference)
                    with set_forward_context(None, config, num_tokens=count):
                        actual = module(inputs.clone()).clone()
                        repeated = module(inputs.clone()).clone()
                        reference_native = reference(inputs.clone()).clone()
                    torch.testing.assert_close(actual.float(), expected.float(), atol=0.05, rtol=0.05)
                    torch.testing.assert_close(actual, repeated, atol=0, rtol=0)
                    if index == 0:
                        torch.testing.assert_close(actual, reference_native, atol=0, rtol=0)
                    case = {
                        "tokens": count,
                        "layer": index,
                        "fp32_reference_max": maximum(actual, expected),
                        "same_route_repeat_max": maximum(actual, repeated),
                        "bf16_native_difference_max": maximum(actual, reference_native),
                    }
                    if count == 128:
                        active = inputs.clone()
                        with set_forward_context(None, config, num_tokens=count):
                            module(active)
                            torch.cuda.synchronize()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                captured = module(active)
                            graph.replay()
                            torch.cuda.synchronize()
                            first = captured.clone()
                            graph.replay()
                            torch.cuda.synchronize()
                        torch.testing.assert_close(actual, captured, atol=0, rtol=0)
                        torch.testing.assert_close(first, captured, atol=0, rtol=0)
                        case["graph_eager_max"] = maximum(actual, captured)
                        case["graph_repeat_max"] = maximum(first, captured)
                    record["cases"].append(case)
                    output.write_text(json.dumps(record, indent=2) + "\n")
            record |= {"status": "passed", "finished_unix": time.time()}
            output.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps({"status": record["status"], "layers": record["layers"], "cases": len(record["cases"])}))
    finally:
        cleanup_dist_env_and_memory()
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


if __name__ == "__main__":
    main()
