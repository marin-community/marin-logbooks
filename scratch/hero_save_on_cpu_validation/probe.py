# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce heterogeneous pinned-cache reuse through actual save_on_cpu hooks."""

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import resource
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

BASE_CONFIG = "expandable_segments:True"
OPTIONS = {
    "baseline": "",
    "threshold_only": ",pinned_max_round_threshold_mb:1",
    "paired": ",pinned_max_round_threshold_mb:1,pinned_max_cached_size_mb:1",
    "cache_cap_only": ",pinned_max_cached_size_mb:1",
}
HOST_KEYS = (
    "allocated_bytes.current",
    "allocated_bytes.peak",
    "active_bytes.current",
    "active_bytes.peak",
    "allocations.current",
    "active_requests.current",
    "num_host_alloc",
    "num_host_free",
)
LIMIT = 2 * 1024**3
MIB = 1024**2


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(temporary, path)


def stats():
    native = torch.cuda.host_memory_stats()
    return {key: int(native[key]) for key in HOST_KEYS}


def sample(receipt, output, phase):
    row = {
        "phase": phase,
        **stats(),
        "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    receipt["observations"].append(row)
    write(output, receipt)
    return row


def settle():
    torch.cuda.synchronize()
    gc.collect()
    poll = torch.empty(1, dtype=torch.uint8, pin_memory=True)
    del poll


def check_values(tensor, value):
    torch.cuda.synchronize()
    if not torch.all(tensor == value).item():
        raise ValueError("Actual H2D/D2H or saved-tensor payload differs")


class CopyEvidence(TorchDispatchMode):
    def __init__(self, receipt, output, phase):
        super().__init__()
        self.receipt, self.output, self.phase = receipt, output, phase

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function is torch.ops.aten.copy_.default and args[0].device.type == "cpu" and args[1].is_cuda:
            destination, source = args[:2]
            self.receipt["copies"].append(
                {
                    "phase": self.phase,
                    "destination_pointer": destination.data_ptr(),
                    "source_shape": list(source.shape),
                    "source_stride": list(source.stride()),
                    "source_storage_offset": source.storage_offset(),
                    "dtype": str(source.dtype),
                    "requested_bytes": destination.numel() * destination.element_size(),
                    "destination_pinned": destination.is_pinned(),
                    "native_before_copy": stats(),
                }
            )
            write(self.output, self.receipt)
            try:
                return function(*args, **(kwargs or {}))
            except torch.AcceleratorError as error:
                self.receipt["failed_copy"] = {
                    **self.receipt["copies"][-1],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                write(self.output, self.receipt)
                raise
        return function(*args, **(kwargs or {}))


def expected_copy_failure(case, receipt, error):
    failure = receipt.get("failed_copy")
    return (
        case == "threshold_only"
        and failure is not None
        and failure["phase"] == "heterogeneous_autograd_save_on_cpu"
        and failure["requested_bytes"] == 48 * MIB
        and failure["destination_pointer"] == receipt.get("prime_pointer")
        and failure["error_type"] == type(error).__name__
        and failure["error"] == str(error)
        and "invalid argument" in str(error)
    )


class SaveSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.save_for_backward(tensor)
        return tensor.square()

    @staticmethod
    def backward(ctx, gradient):
        (tensor,) = ctx.saved_tensors
        return gradient * 2 * tensor


def source_evidence():
    graph = Path(inspect.getfile(torch.autograd.graph.save_on_cpu))
    include = Path(torch.__file__).parent / "include"
    sources = {}
    for name, path, anchors in (
        ("graph", graph, ("def pack_to_cpu", "packed.copy_")),
        (
            "host_header",
            include / "ATen/core/CachingHostAllocator.h",
            ("get_free_block(size_t", "maybe_cache_block(", "roundSize"),
        ),
        ("cuda_host_header", include / "ATen/cuda/CachingHostAllocator.h", ("CachingHostAllocator",)),
    ):
        lines = path.read_text().splitlines()
        snippets = []
        for index, line in enumerate(lines):
            if any(anchor in line for anchor in anchors):
                snippets.append({"line": index + 1, "text": "\n".join(lines[index : index + 14])})
        sources[name] = {"path": str(path), "sha256": digest(path), "snippets": snippets[:6]}
    native = [
        Path(torch._C.__file__),
        *(Path(torch.__file__).parent / "lib").glob("lib*c10_cuda*.so"),
        Path(torch.__file__).parent / "lib/libtorch_cuda.so",
    ]
    return {
        "torch_git_version": torch.version.git_version,
        "installed_sources": sources,
        "native_binary_sha256": {str(path): digest(path) for path in native},
    }


def storage_cycles(args, receipt):
    spec = importlib.util.spec_from_file_location("frozen_grug_muonh", args.optimizer_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parameter = torch.nn.Parameter(torch.full((3072, 3072), 0.25, device="cuda", dtype=torch.float32))
    optimizer = module.MegatronGrugMuonH(
        [{"params": [parameter], "optimizer": "grug_muonh"}],
        lr=1e-6,
        betas=(0.9, 0.95),
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        eps=1e-8,
        muon_eps=1e-8,
        qkv_split_shapes=(1, 1, 1),
        offload_momentum=True,
    )
    optimizer.initialize_state()
    momentum = optimizer.state[parameter]["momentum_buffer"]
    pointer = momentum.data_ptr()
    for cycle in (1, 2):
        parameter.data = parameter.data.to("cpu", non_blocking=True)
        settle()
        check_values(parameter, cycle - 0.75)
        sample(receipt, args.output, f"cycle{cycle}_offloaded")
        parameter.data = parameter.data.to("cuda", non_blocking=True)
        settle()
        check_values(parameter, cycle - 0.75)
        with torch.no_grad():
            parameter.add_(1)
            momentum.add_(1)
        if not momentum.is_pinned() or momentum.data_ptr() != pointer:
            raise ValueError("Live MuonH momentum changed identity/pinning")
        check_values(momentum, cycle)
        sample(receipt, args.output, f"cycle{cycle}_backloaded")
    receipt["completed_storage_cycles"] = 2
    # Keep all three objects alive throughout activation tests.
    return parameter, optimizer, momentum


def small_reuse(args, receipt):
    torch.accelerator.memory.empty_host_cache()
    before = sample(receipt, args.output, "below_cutoff_before")
    small = torch.empty((257, 257), dtype=torch.float32, pin_memory=True)
    pointer = small.data_ptr()
    first = sample(receipt, args.output, "below_cutoff_prime_live")
    if first["active_bytes.current"] - before["active_bytes.current"] != 512 * 1024:
        raise ValueError("Below-cutoff prime was not rounded to its512KiB block")
    del small
    larger = torch.empty((300, 300), dtype=torch.float32, pin_memory=True)
    second = sample(receipt, args.output, "below_cutoff_larger_live")
    if larger.data_ptr() != pointer or first["allocated_bytes.current"] != second["allocated_bytes.current"]:
        raise ValueError("Below-cutoff power-of-two cache reuse was not observed")
    if first["active_bytes.current"] != second["active_bytes.current"]:
        raise ValueError("Below-cutoff requests were not rounded to the same block")
    receipt["below_cutoff_reuse"] = {"pointer": pointer, "requested_bytes": [257 * 257 * 4, 300 * 300 * 4]}
    del larger


def activation_cases(args, receipt):
    # Optional managed MCore dependency is imported only for actual GPU execution.
    from megatron.core import tensor_parallel  # noqa: PLC0415

    torch.accelerator.memory.empty_host_cache()
    before_prime = sample(receipt, args.output, "before_prime_activation")
    hook = torch.autograd.graph.save_on_cpu(pin_memory=True)
    prime = torch.full((3072, 1, 6144), 0.25, dtype=torch.bfloat16, device="cuda")
    with CopyEvidence(receipt, args.output, "prime_save_on_cpu"):
        packed = hook.pack_hook(prime)
    settle()
    check_values(packed[1], 0.25)
    receipt["prime_pointer"] = packed[1].data_ptr()
    receipt["checkpoint_source_sha256"] = digest(inspect.getfile(tensor_parallel.checkpoint))
    receipt["prime_owned_stats"] = sample(receipt, args.output, "prime_activation_live")
    prime_block = 64 * MIB if args.case == "baseline" else 36 * MIB
    if receipt["prime_owned_stats"]["active_bytes.current"] - before_prime["active_bytes.current"] != prime_block:
        raise ValueError("Above-cutoff prime native block size differs from requested configuration")
    del packed
    settle()
    sample(receipt, args.output, "prime_activation_freed")
    # BF16 hidden6144, microbatch1:36MiB->48MiB share the default64MiB bucket.
    target = torch.full((4096, 1, 6144), 0.25, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    with CopyEvidence(receipt, args.output, "heterogeneous_autograd_save_on_cpu"):
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            result = tensor_parallel.checkpoint(lambda value: value.square(), False, target)
        result.sum().backward()
    settle()
    check_values(target.grad, 0.5)
    # These geometry/stride variants are derived from the Hero config, not
    # claimed to be the exact unrecorded activation that failed in .35.
    for name, view in (
        ("max_sequence", torch.full((4224, 1, 6144), 0.25, dtype=torch.bfloat16, device="cuda")),
        ("transpose", target.detach().transpose(0, 2)),
        ("offset_slice", torch.full((4225, 1, 6144), 0.25, dtype=torch.bfloat16, device="cuda")[1:]),
    ):
        leaf = view.detach().requires_grad_()
        with CopyEvidence(receipt, args.output, name):
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                output = SaveSquare.apply(leaf)
            output.sum().backward()
        settle()
        check_values(leaf.grad, 0.5)
        receipt["autograd_cases"].append(name)
    target_copy = next(item for item in receipt["copies"] if item["phase"] == "heterogeneous_autograd_save_on_cpu")
    if args.case == "baseline":
        if target_copy["destination_pointer"] != receipt["prime_pointer"]:
            raise ValueError("Baseline did not exercise same-bucket cached block reuse")
        if target_copy["native_before_copy"]["num_host_alloc"] != receipt["prime_owned_stats"]["num_host_alloc"]:
            raise ValueError("Baseline grew the host pool instead of reusing the primed block")
    if args.case in ("paired", "cache_cap_only"):
        freed = next(item for item in receipt["observations"] if item["phase"] == "prime_activation_freed")
        if freed["num_host_free"] <= receipt["prime_owned_stats"]["num_host_free"]:
            raise ValueError("Above-cutoff prime block was cached instead of released")
        if target_copy["native_before_copy"]["num_host_alloc"] <= receipt["prime_owned_stats"]["num_host_alloc"]:
            raise ValueError("Above-cutoff target did not get a fresh native allocation")
    receipt["heterogeneous_autograd_passed"] = True


def run_case(args):
    if torch.__version__ != "2.13.0+cu132" or torch.version.cuda != "13.2":
        raise ValueError("Requires exact managed Torch2.13.0+cu132")
    if platform.machine() != "aarch64" or torch.cuda.device_count() != 1 or "GB200" not in torch.cuda.get_device_name():
        raise ValueError("Requires one visible ARM GB200")
    if digest(args.optimizer_source) != args.optimizer_sha256:
        raise ValueError("Frozen MuonH source changed")
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != BASE_CONFIG + OPTIONS[args.case]:
        raise ValueError("Fresh process configuration does not match case")
    receipt = {
        "result": "running",
        "case": args.case,
        "torch": torch.__version__,
        "allocator_config": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
        "hardware": {
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "machine": platform.machine(),
        },
        "sources": source_evidence(),
        "optimizer_sha256": args.optimizer_sha256,
        "probe_sha256": digest(__file__),
        "observations": [],
        "copies": [],
        "autograd_cases": [],
    }
    sample(receipt, args.output, "initial")
    try:
        parameter, optimizer, momentum = storage_cycles(args, receipt)
        momentum_pointer = momentum.data_ptr()
        small_reuse(args, receipt)
        activation_cases(args, receipt)
        check_values(momentum, 2)
        if momentum.data_ptr() != momentum_pointer or optimizer.state[parameter]["momentum_buffer"] is not momentum:
            raise ValueError("Activation offload changed live MuonH momentum")
        if args.case == "threshold_only":
            raise ValueError("Threshold-only unexpectedly passed; hazard not reproduced")
        receipt["result"] = "passed"
    except torch.AcceleratorError as error:
        receipt.update(result="failed", error_type=type(error).__name__, error=str(error)[:2000])
        if not expected_copy_failure(args.case, receipt, error):
            write(args.output, receipt)
            raise
        receipt["result"] = "expected_allocator_failure"
        receipt["undersized_block_reuse_before_invalid_copy"] = True
        receipt["live_momentum_preserved_before_failure"] = momentum.data_ptr() == momentum_pointer
        # Do not reuse a CUDA context after its expected failed copy.
    except Exception as error:
        receipt.update(result="failed", error_type=type(error).__name__, error=str(error)[:2000])
        write(args.output, receipt)
        raise
    # Native per-phase current bytes plus GPU allocator peak bound test tensors;
    # process RSS/imported libraries remain a separate64GiB pod budget.
    host_peak = stats()["allocated_bytes.peak"]
    gpu_peak = torch.cuda.max_memory_allocated()
    receipt["allocation_bounds"] = {
        "host_owned_observed_peak_bytes": host_peak,
        "gpu_peak_bytes": gpu_peak,
        "limit_bytes": LIMIT,
    }
    if host_peak >= LIMIT or gpu_peak >= LIMIT:
        receipt["result"] = "failed"
        write(args.output, receipt)
        raise ValueError("Allocator test exceeds the2GiB tensor bound")
    write(args.output, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=(*OPTIONS, "all"), required=True)
    parser.add_argument("--optimizer-source", type=Path, required=True)
    parser.add_argument("--optimizer-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.plan:
        print(
            json.dumps(
                {"cases": list(OPTIONS), "prime_bytes": 36 * MIB, "target_bytes": 48 * MIB, "gpu_execution": "none"},
                indent=2,
            )
        )
        return
    if args.case != "all":
        run_case(args)
        return
    args.output.mkdir(parents=True)
    identity = None
    for case, option in OPTIONS.items():
        environment = dict(os.environ)
        environment.pop("PYTORCH_ALLOC_CONF", None)
        environment["PYTORCH_CUDA_ALLOC_CONF"] = BASE_CONFIG + option
        destination = args.output / f"{case}.json"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--case",
                case,
                "--optimizer-source",
                str(args.optimizer_source.resolve()),
                "--optimizer-sha256",
                args.optimizer_sha256,
                "--output",
                str(destination.resolve()),
            ],
            env=environment,
            check=True,
            timeout=240,
        )
        result = json.loads(destination.read_text())
        installed_identity = {"sources": result["sources"], "checkpoint_sha256": result["checkpoint_source_sha256"]}
        if identity is None:
            identity = installed_identity
        elif installed_identity != identity:
            raise ValueError("Installed Torch/MCore source or native binary identity changed between fresh cases")
        expected = "expected_allocator_failure" if case == "threshold_only" else "passed"
        if result["result"] != expected:
            raise ValueError(f"{case} did not produce the required discriminating result")
    print(json.dumps({"result": "passed", "fresh_process_cases": 4, "output": str(args.output)}))


if __name__ == "__main__":
    main()
