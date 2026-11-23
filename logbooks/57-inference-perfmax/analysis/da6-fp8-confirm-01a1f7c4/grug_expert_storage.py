# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record device storage, arithmetic mode and typed cache views on each DP rank."""

import hashlib
import json
import os
import traceback
import urllib.error
import urllib.request
from fnmatch import fnmatchcase
from pathlib import Path


def storage(tensor):
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "bytes": tensor.numel() * tensor.element_size(),
    }


def cache_views(value, tensor_type):
    if isinstance(value, tensor_type) and value.numel():
        yield value
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from cache_views(child, tensor_type)


class GrugExpertStorage:
    def inspect_grug_experts(self, scheme: str, backend: str, ignored: list[str], group: str) -> dict:
        # These import the exact vLLM under test, in its GPU worker process.
        import torch  # noqa: PLC0415
        from vllm.model_executor.layers.fused_moe import RoutedExperts  # noqa: PLC0415

        model = self.get_model()
        config = self.vllm_config
        parallel = config.parallel_config
        hf = config.model_config.hf_config
        routed, caches, errors = {}, {}, []
        for name, module in model.named_modules():
            if not isinstance(module, RoutedExperts):
                continue
            method = module.quant_method
            quant = getattr(method, "moe_quant_config", None)
            weights = [module.w13_weight, module.w2_weight]
            activation_dtype = getattr(quant, "quant_dtype", None)
            weight_dtype = getattr(quant, "weight_quant_dtype", None)
            selected_backend = getattr(method, "fp8_backend", None)
            selected_backend = getattr(selected_backend, "value", None)
            prefix = name.removesuffix(".routed_experts")
            expected_fp8 = scheme != "bf16" and not any(fnmatchcase(prefix, pattern) for pattern in ignored)
            if bool(selected_backend) != expected_fp8:
                errors.append(f"{name} does not match the requested precision")
            block_shape = getattr(module, "weight_block_size", None)
            fp8_storage_dtype = torch.int32 if backend == "MARLIN" else torch.float8_e4m3fn
            expected_dtype = fp8_storage_dtype if selected_backend else torch.bfloat16
            local_experts = hf.num_experts // parallel.data_parallel_size
            logical_bytes = local_experts * 3 * module.hidden_size * module.intermediate_size_per_partition
            expected_bytes = logical_bytes * (2 if selected_backend is None else 1)
            item = {
                "method": type(method).__name__,
                "backend": selected_backend,
                "weight_dtype": str(weight_dtype),
                "activation_dtype": str(activation_dtype or config.model_config.dtype),
                "block_shape": block_shape,
                "expert_count": module.w13_weight.shape[0],
                "weights": [storage(tensor) for tensor in weights],
            }
            scale_name = "weight_scale_inv" if scheme == "fp8_per_block" else "weight_scale"
            scales = [] if selected_backend is None else [
                storage(getattr(module, f"{name}_{scale_name}")) for name in ("w13", "w2")
            ]
            item["scales"] = scales
            routed[name] = item
            if selected_backend is not None:
                if (
                    selected_backend != backend
                    or weight_dtype != torch.float8_e4m3fn
                    or activation_dtype != (None if backend == "MARLIN" else torch.float8_e4m3fn)
                    or block_shape != ([128, 128] if scheme == "fp8_per_block" else None)
                ):
                    errors.append(f"{name} uses unexpected FP8 arithmetic")
            if (
                any(tensor.dtype != expected_dtype or not tensor.is_cuda for tensor in weights)
                or any(tensor.shape[0] != local_experts for tensor in weights)
                or sum(weight["bytes"] for weight in item["weights"]) != expected_bytes
            ):
                errors.append(f"{name} uses unexpected expert storage")
        for name, layer in config.compilation_config.static_forward_context.items():
            tensors = list(cache_views(getattr(layer, "kv_cache", None), torch.Tensor))
            if tensors:
                caches[name] = {
                    "kind": "short_conv" if type(layer).__name__ == "GrugMoeShortConv" else "attention",
                    "views": [storage(tensor) for tensor in tensors],
                }
            if any(tensor.dtype != torch.bfloat16 or not tensor.is_cuda for tensor in tensors):
                errors.append(f"{name} has a cache view outside BF16")
        fp8_layers = [name for name, item in routed.items() if item["backend"]]
        attention_layers = sum(item["kind"] == "attention" for item in caches.values())
        short_conv_layers = len(caches) - attention_layers
        if (
            len(routed) != hf.num_hidden_layers
            or attention_layers != hf.num_hidden_layers
            or short_conv_layers != hf.num_hidden_layers * (3 if getattr(hf, "sconv", False) else 0)
            or parallel.tensor_parallel_size != 1
        ):
            errors.append("Incomplete layer, cache or TP1 coverage")
        main_bytes, seen = 0, set()
        for name, tensor in model.named_parameters():
            expected = torch.float32 if ".mlp.router." in name else torch.bfloat16
            if ".experts." not in name and tensor.dtype != expected:
                errors.append(f"{name} has dtype {tensor.dtype} instead of {expected}")
            device_storage = tensor.untyped_storage()
            address = device_storage.data_ptr()
            if address not in seen:
                main_bytes += device_storage.nbytes()
                seen.add(address)
        rank = parallel.data_parallel_rank
        out = Path(os.environ["IRIS_OUTPUT_DIR"]) / group / "weight-storage"
        out.mkdir(parents=True, exist_ok=True)
        record = {
            "status": "failed" if errors else "passed",
            "extension_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "errors": errors,
            "dp_rank": rank,
            "dp_size": parallel.data_parallel_size,
            "fp8_layers": fp8_layers,
            "expert_weight_bytes": sum(weight["bytes"] for row in routed.values() for weight in row["weights"]),
            "model_parameter_storage_bytes": main_bytes,
            "kv_cache_views": caches,
            "routed_experts": routed,
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        }
        (out / f"dp-{rank}.json").write_text(json.dumps(record, indent=2) + "\n")
        if errors:
            raise ValueError("; ".join(errors[:10]))
        # Workers keep full native records. The DP client returns just rank 0.
        return {key: value for key, value in record.items() if key not in {"kv_cache_views", "routed_experts"}}


def main():
    profile = json.loads(Path(os.environ["KNOTMETER_PROFILE"]).read_text())
    output = Path(os.environ["KNOTMETER_CHECK_OUTPUT"])
    base = os.environ["KNOTMETER_SERVER_URL"].removesuffix("/v1")
    args = profile["serving"]["extra_args"]
    selector = json.loads(args[args.index("--quantization-config") + 1]) if profile["quantization"] else {}
    scheme = selector.get("moe", "bf16")
    backend = json.loads(args[args.index("--kernel-config") + 1])["moe_backend"].upper()
    ignored = selector.get("ignore", [])
    request = urllib.request.Request(
        f"{base}/collective_rpc",
        data=json.dumps(
            {"method": "inspect_grug_experts", "args": [scheme, backend, ignored, output.parent.name], "timeout": 120}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"status": "failed", "runtime_source": os.environ["KNOTMETER_RUNTIME_SOURCE"], "scheme": scheme}
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            reports = json.load(response)["results"]
        report["workers_returned"] = reports
        expected_layers = 26 if profile["gpus"]["type"] == "H100" else 48
        expected_fp8 = sum(
            scheme != "bf16" and not any(fnmatchcase(f"model.layers.{layer}.mlp.experts", pattern) for pattern in ignored)
            for layer in range(expected_layers)
        )
        if reports and all(
            worker["status"] == "passed"
            and worker["dp_size"] == profile["gpus"]["count"]
            and worker["extension_sha256"] == digest
            and len(worker["fp8_layers"]) == expected_fp8
            for worker in reports
        ):
            report["status"] = "passed"
    except urllib.error.HTTPError as error:
        report["error"] = error.read().decode(errors="replace")
        raise
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "passed":
        raise SystemExit("Loaded storage and arithmetic did not match the trial")


if __name__ == "__main__":
    main()
