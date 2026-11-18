# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect native tensor metadata without copying or scoring tensor values."""

import hashlib
import json
import os
from pathlib import Path


def describe_tensor(tensor) -> dict:
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "element_size": tensor.element_size(),
        "payload_bytes": tensor.numel() * tensor.element_size(),
        "storage_bytes": tensor.untyped_storage().nbytes(),
    }


def check_packed_fp8(info: dict, logical_elements: int) -> bool:
    # Native pack_fp8_to_int32 views four E4M3FN bytes as one int32, then
    # gptq_marlin_repack(..., num_bits=8) preserves the byte count.
    return (
        info["dtype"] == "torch.int32"
        and info["element_size"] == 4
        and info["payload_bytes"] == logical_elements
        and info["storage_bytes"] == logical_elements
    )


def cache_tensors(value):
    if hasattr(value, "dtype") and hasattr(value, "numel"):
        if value.numel():
            yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from cache_tensors(item)


class HeroWeightMetadata:
    def inspect_hero_storage(self, expected_mode: str, output_group: str = "point") -> dict:
        # Imports resolve against the installed wheel. This extension is an
        # additional native worker method; it never replaces a model module.
        from vllm.model_executor.layers.fused_moe import RoutedExperts  # noqa: PLC0415
        from vllm.model_executor.layers.linear import LinearBase  # noqa: PLC0415

        if expected_mode != "selective-fp8-marlin":
            raise ValueError("This campaign metadata check supports only selective per-tensor FP8 Marlin")
        if output_group not in {"point", "correctness"}:
            raise ValueError("Unknown campaign artifact directory")
        config = self.vllm_config
        parallel = config.parallel_config
        hf = config.model_config.hf_config
        errors, experts, projections, routers, caches = [], [], [], [], []
        for name, module in self.get_model().named_modules():
            if isinstance(module, RoutedExperts):
                method = module.quant_method
                quant = getattr(method, "moe_quant_config", None)
                hidden = module.hidden_size
                intermediate = module.intermediate_size_per_partition
                local_experts = module.w13_weight.shape[0]
                weights = {key: describe_tensor(getattr(module, key)) for key in ("w13_weight", "w2_weight")}
                scales = {key: describe_tensor(getattr(module, key)) for key in ("w13_weight_scale", "w2_weight_scale")}
                entry = {
                    "name": name,
                    "method": type(method).__name__,
                    "backend": str(getattr(method, "fp8_backend", None)),
                    "consumer_weight_dtype": str(getattr(quant, "weight_quant_dtype", None)),
                    "consumer_activation_dtype": str(getattr(quant, "quant_dtype", None)),
                    "hidden": hidden,
                    "intermediate": intermediate,
                    "local_experts": local_experts,
                    "weights": weights,
                    "scales": scales,
                }
                expected_local = hf.num_experts // parallel.data_parallel_size
                good = (
                    entry["method"] == "Fp8PerTensorOnlineMoEMethod"
                    and entry["backend"].endswith("MARLIN")
                    and entry["consumer_weight_dtype"] == "torch.float8_e4m3fn"
                    and entry["consumer_activation_dtype"] == "None"
                    and local_experts == expected_local
                    and module.w2_weight.shape[0] == expected_local
                    and check_packed_fp8(weights["w13_weight"], local_experts * 2 * intermediate * hidden)
                    and check_packed_fp8(weights["w2_weight"], local_experts * intermediate * hidden)
                    and scales["w13_weight_scale"]["numel"] == local_experts * 2 * intermediate
                    and scales["w2_weight_scale"]["numel"] == local_experts * hidden
                    and all(item["device"].startswith("cuda") for item in weights.values())
                )
                entry["status"] = "passed" if good else "failed"
                experts.append(entry)
                if not good:
                    errors.append(name + ": expert storage/consumer metadata differs")
            if isinstance(module, LinearBase):
                item = {"name": name, "weight": describe_tensor(module.weight)}
                if name.endswith(".router"):
                    item["bias"] = describe_tensor(module.bias)
                    routers.append(item)
                    if item["weight"]["dtype"] != "torch.float32" or item["bias"]["dtype"] != "torch.float32":
                        errors.append(name + ": router is not FP32")
                else:
                    projections.append(item)
                    if item["weight"]["dtype"] != "torch.bfloat16":
                        errors.append(name + ": projection is not BF16")
                if not item["weight"]["device"].startswith("cuda"):
                    errors.append(name + ": weight is not CUDA-resident")
        for name, layer in config.compilation_config.static_forward_context.items():
            for index, tensor in enumerate(cache_tensors(getattr(layer, "kv_cache", None))):
                item = {
                    "name": name,
                    "index": index,
                    "kind": "short_conv_state" if type(layer).__name__ == "GrugMoeShortConv" else "attention_kv",
                    **describe_tensor(tensor),
                }
                caches.append(item)
                if item["dtype"] != "torch.bfloat16" or not item["device"].startswith("cuda"):
                    errors.append(name + ": cache/state view is not CUDA BF16")
        if len(experts) != hf.num_hidden_layers or len(routers) != hf.num_hidden_layers or not projections:
            errors.append("Full layer coverage is incomplete")
        if not all(any(item["kind"] == kind for item in caches) for kind in ("attention_kv", "short_conv_state")):
            errors.append("Typed attention/state-cache views are missing")
        record = {
            "status": "failed" if errors else "passed",
            "errors": errors,
            "scope": "Metadata only: dtype, shape and storage; no tensor values copied or evaluated",
            "dp_rank": parallel.data_parallel_rank,
            "dp_size": parallel.data_parallel_size,
            "tp_size": parallel.tensor_parallel_size,
            "extension_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "experts": experts,
            "projections": projections,
            "routers": routers,
            "cache_precision": {
                "requested_attention": config.cache_config.cache_dtype,
                "requested_mamba": config.cache_config.mamba_cache_dtype,
                "model_dtype": str(config.model_config.dtype),
                "typed_views": caches,
                "note": "Backing storage can be shared; these per-view storage sizes must not be added together",
            },
        }
        path = (
            Path(os.environ["IRIS_OUTPUT_DIR"])
            / output_group
            / "weight-storage"
            / f"dp-{parallel.data_parallel_rank}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n")
        # Pinned DPLBAsyncMPClient.call_utility_async broadcasts to every DP
        # engine but returns only the first result. A failure from any rank
        # propagates through gather; native utility failures do not stop workers.
        if errors:
            raise ValueError("; ".join(errors[:5]))
        return record
