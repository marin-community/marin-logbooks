# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect native BF16 Triton and packed FP8 Marlin experts with separate KV views."""

import hashlib
import json
import os
import re
from pathlib import Path

from hero_weight_metadata import cache_tensors, check_packed_fp8, describe_tensor


class HeroHybridMetadata:
    def inspect_hero_hybrid_storage(self, bf16_layers: list[int], output_group: str = "point") -> dict:
        from vllm.model_executor.layers.fused_moe import RoutedExperts  # noqa: PLC0415
        from vllm.model_executor.layers.linear import LinearBase  # noqa: PLC0415

        config = self.vllm_config
        hf, parallel = config.model_config.hf_config, config.parallel_config
        if output_group not in {"point", "correctness"}:
            raise ValueError("Unknown campaign artifact directory")
        if sorted(set(bf16_layers)) != bf16_layers or any(i < 0 or i >= hf.num_hidden_layers for i in bf16_layers):
            raise ValueError("BF16 layer indices must be unique, sorted and in this model")
        errors, experts, projections, routers, caches = [], [], [], [], []
        for name, module in self.get_model().named_modules():
            if isinstance(module, RoutedExperts):
                match = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.experts\.routed_experts", name)
                if match is None:
                    raise ValueError("Unexpected expert module: " + name)
                index = int(match[1])
                bf16 = index in bf16_layers
                method = module.quant_method
                quant = method.moe_quant_config
                hidden, intermediate = module.hidden_size, module.intermediate_size_per_partition
                local = hf.num_experts // parallel.data_parallel_size
                weights = {key: describe_tensor(getattr(module, key)) for key in ("w13_weight", "w2_weight")}
                scales = {
                    key: describe_tensor(getattr(module, key))
                    for key in ("w13_weight_scale", "w2_weight_scale")
                    if hasattr(module, key)
                }
                entry = {
                    "name": name,
                    "layer": index,
                    "expected_precision": "bf16" if bf16 else "fp8",
                    "method": type(method).__name__,
                    "backend": str(getattr(method, "unquantized_backend" if bf16 else "fp8_backend", None)),
                    "consumer_weight_dtype": str(quant.weight_quant_dtype),
                    "consumer_activation_dtype": str(quant.quant_dtype),
                    "weights": weights,
                    "scales": scales,
                    "hidden": hidden,
                    "intermediate": intermediate,
                    "local_experts": local,
                }
                dtype = "torch.bfloat16" if bf16 else "torch.float8_e4m3fn"
                elements = {"w13_weight": local * 2 * intermediate * hidden, "w2_weight": local * hidden * intermediate}
                if bf16:
                    storage_good = (
                        weights["w13_weight"]["shape"] == [local, 2 * intermediate, hidden]
                        and weights["w2_weight"]["shape"] == [local, hidden, intermediate]
                        and not scales
                        and all(
                            info["dtype"] == "torch.bfloat16"
                            and info["device"].startswith("cuda")
                            and info["numel"] == elements[key]
                            and info["element_size"] == 2
                            and info["payload_bytes"] == info["storage_bytes"] == elements[key] * 2
                            for key, info in weights.items()
                        )
                    )
                else:
                    storage_good = (
                        all(
                            info["shape"][0] == local
                            and info["device"].startswith("cuda")
                            and check_packed_fp8(info, elements[key])
                            for key, info in weights.items()
                        )
                        and len(scales) == 2
                        and scales["w13_weight_scale"]["numel"] == local * 2 * intermediate
                        and scales["w2_weight_scale"]["numel"] == local * hidden
                        and all(
                            info["dtype"] == "torch.bfloat16" and info["device"].startswith("cuda")
                            for info in scales.values()
                        )
                    )
                good = (
                    entry["method"] == ("UnquantizedFusedMoEMethod" if bf16 else "Fp8PerTensorOnlineMoEMethod")
                    and entry["backend"].endswith("TRITON" if bf16 else "MARLIN")
                    and entry["consumer_weight_dtype"] == ("None" if bf16 else dtype)
                    and entry["consumer_activation_dtype"] == "None"
                    and storage_good
                )
                entry["status"] = "passed" if good else "failed"
                experts.append(entry)
                if not good:
                    errors.append(name + ": actual expert precision/layout differs")
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
                    errors.append(name + ": weight is not CUDA resident")
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
        if sorted(item["layer"] for item in experts) != list(range(hf.num_hidden_layers)):
            errors.append("Full expert layer coverage is incomplete")
        if len(routers) != hf.num_hidden_layers or not projections:
            errors.append("Projection/router coverage is incomplete")
        if not all(any(item["kind"] == kind for item in caches) for kind in ("attention_kv", "short_conv_state")):
            errors.append("Typed attention/state-cache views are missing")
        record = {
            "status": "failed" if errors else "passed",
            "errors": errors,
            "scope": "Metadata only; no tensor values are copied or scored",
            "dp_rank": parallel.data_parallel_rank,
            "dp_size": parallel.data_parallel_size,
            "tp_size": parallel.tensor_parallel_size,
            "extension_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "helper_sha256": (
                hashlib.sha256(Path(__file__).with_name("hero_weight_metadata.py").read_bytes()).hexdigest()
            ),
            "expected_bf16_layers": bf16_layers,
            "bf16_expert_layers": sum(item["expected_precision"] == "bf16" for item in experts),
            "fp8_expert_layers": sum(item["expected_precision"] == "fp8" for item in experts),
            "expert_payload_bytes": {
                precision: sum(
                    info["payload_bytes"]
                    for item in experts
                    if item["expected_precision"] == precision
                    for info in item["weights"].values()
                )
                for precision in ("bf16", "fp8")
            },
            "experts": experts,
            "projections": projections,
            "routers": routers,
            "cache_precision": {
                "requested_attention": config.cache_config.cache_dtype,
                "requested_mamba": config.cache_config.mamba_cache_dtype,
                "model_dtype": str(config.model_config.dtype),
                "typed_views": caches,
                "note": "Views may share backing storage; do not add these storage sizes",
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
        if errors:
            raise ValueError("; ".join(errors[:5]))
        return record
