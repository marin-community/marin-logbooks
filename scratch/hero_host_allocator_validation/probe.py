# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure exact Torch host-pinned storage lifetimes without training a model."""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
from pathlib import Path

import torch

TORCH_VERSION = "2.13.0+cu132"
SHAPE = (3072, 3072)
RAW_BYTES = 3072 * 3072 * 4
DEFAULT_BLOCK_BYTES = 64 * 1024**2
TEST_ALLOCATION_LIMIT_BYTES = 2 * 1024**3
ALLOCATOR_BASE = "expandable_segments:True"
CASES = ("baseline", "flush", "threshold")
TIMINGS = ("eager", "first_step")
STATS_KEYS = (
    "allocated_bytes.current",
    "allocated_bytes.peak",
    "active_bytes.current",
    "active_bytes.peak",
    "allocations.current",
    "active_requests.current",
    "num_host_alloc",
    "num_host_free",
)


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(temporary, path)


def settle_copies() -> None:
    torch.cuda.synchronize()
    gc.collect()
    # Host stats do not poll completed stream events. A one-byte allocation
    # exercises the allocator's normal event-processing path in every case.
    event_poll = torch.empty(1, dtype=torch.uint8, pin_memory=True)
    del event_poll


def snapshot(receipt: dict, output: Path, phase: str) -> dict:
    settle_copies()
    stats = torch.cuda.host_memory_stats()
    selected = {key: int(stats[key]) for key in STATS_KEYS}
    selected.update(
        phase=phase,
        cache_gap_bytes=selected["allocated_bytes.current"] - selected["active_bytes.current"],
        gpu_allocated_bytes=torch.cuda.memory_allocated(),
        gpu_reserved_bytes=torch.cuda.memory_reserved(),
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    receipt["observations"].append(selected)
    write_receipt(output, receipt)
    return selected


def optimizer_type(path: Path, expected_sha256: str):
    if digest(path) != expected_sha256:
        raise ValueError("MuonH source differs from the specified immutable source")
    spec = importlib.util.spec_from_file_location("hero_probe_grug_muonh", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MegatronGrugMuonH


def verify_parameter(parameter, expected: float) -> None:
    torch.cuda.synchronize()
    if not torch.all(parameter == expected).item():
        raise ValueError("Parameter payload changed during the real H2D/D2H roundtrip")


def run_case(args) -> None:
    if torch.__version__ != TORCH_VERSION or torch.version.cuda != "13.2":
        raise ValueError(f"Probe requires exact {TORCH_VERSION}, got {torch.__version__}/{torch.version.cuda}")
    if platform.machine() != "aarch64" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Probe requires one visible ARM CUDA device")
    if "GB200" not in torch.cuda.get_device_name():
        raise ValueError("Probe requires the requested GB200 device")
    expected_config = ALLOCATOR_BASE + (",pinned_max_round_threshold_mb:1" if args.case == "threshold" else "")
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != expected_config:
        raise ValueError("Fresh child allocator configuration differs from its requested case")
    expected_block = RAW_BYTES if args.case == "threshold" else DEFAULT_BLOCK_BYTES
    receipt = {
        "result": "running",
        "scope": (
            "Actual MuonH state initialization and master-storage roundtrips; "
            "no optimizer update or full-model fit claim"
        ),
        "case": args.case,
        "momentum_timing": args.momentum_timing,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "allocator_config": expected_config,
        "parameter_shape": list(SHAPE),
        "raw_tensor_bytes": RAW_BYTES,
        "expected_pinned_block_bytes": expected_block,
        "optimizer_source": str(args.optimizer_source),
        "optimizer_source_sha256": args.optimizer_sha256,
        "probe_sha256": digest(Path(__file__)),
        "observations": [],
    }
    write_receipt(args.output, receipt)
    try:
        cls = optimizer_type(args.optimizer_source, args.optimizer_sha256)
        parameter = torch.nn.Parameter(torch.full(SHAPE, 0.25, device="cuda", dtype=torch.float32))
        optimizer = cls(
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
        initial = snapshot(receipt, args.output, "initial")
        momentum = None
        momentum_pointer = None
        if args.momentum_timing == "eager":
            optimizer.initialize_state()
            momentum = optimizer.state[parameter]["momentum_buffer"]
            momentum_pointer = momentum.data_ptr()
            initialized = snapshot(receipt, args.output, "momentum_initialized_eager")
            if initialized["active_bytes.current"] - initial["active_bytes.current"] != expected_block:
                raise ValueError("Actual MuonH pinned momentum allocation does not match the expected rounding")
        # Match the startup optimizer offload before the first training backload.
        parameter.data = parameter.data.to("cpu", non_blocking=True)
        torch.cuda.synchronize()
        if not parameter.is_pinned():
            raise ValueError("Nonblocking D2H did not create a pinned master copy")
        verify_parameter(parameter, 0.25)
        startup = snapshot(receipt, args.output, "startup_master_offloaded")
        for cycle in (1, 2):
            parameter.data = parameter.data.to("cuda", non_blocking=True)
            # This synchronization is the existing Megatron strategy boundary.
            torch.cuda.synchronize()
            verify_parameter(parameter, cycle - 0.75)
            before_flush = snapshot(receipt, args.output, f"cycle{cycle}_backloaded_before_flush")
            if args.case == "flush":
                torch.accelerator.memory.empty_host_cache()
            after_flush = snapshot(receipt, args.output, f"cycle{cycle}_backloaded_after_flush")
            if args.case == "flush":
                if before_flush["active_bytes.current"] != after_flush["active_bytes.current"]:
                    raise ValueError("Host cache release changed live pinned allocations")
                if before_flush["allocated_bytes.current"] - after_flush["allocated_bytes.current"] < expected_block:
                    raise ValueError("Host cache release did not reclaim the freed master block")
            if momentum is None:
                before_momentum = snapshot(receipt, args.output, "before_first_step_momentum_init")
                optimizer.initialize_state()
                momentum = optimizer.state[parameter]["momentum_buffer"]
                momentum_pointer = momentum.data_ptr()
                after_momentum = snapshot(receipt, args.output, "after_first_step_momentum_init")
                if after_momentum["active_bytes.current"] - before_momentum["active_bytes.current"] != expected_block:
                    raise ValueError("Lazy MuonH pinned momentum allocation does not match expected rounding")
            if not momentum.is_pinned() or momentum.device.type != "cpu":
                raise ValueError("Actual MuonH momentum did not remain pinned on the CPU")
            with torch.no_grad():
                parameter.add_(1.0)
                momentum.add_(1.0)
            verify_parameter(parameter, cycle + 0.25)
            parameter.data = parameter.data.to("cpu", non_blocking=True)
            torch.cuda.synchronize()
            verify_parameter(parameter, cycle + 0.25)
            if not parameter.is_pinned() or momentum.data_ptr() != momentum_pointer:
                raise ValueError("Master pinning or live momentum identity changed")
            if optimizer.state[parameter]["momentum_buffer"] is not momentum or not torch.all(momentum == cycle).item():
                raise ValueError("Live MuonH momentum payload changed across roundtrips/cache release")
            offloaded = snapshot(receipt, args.output, f"cycle{cycle}_master_and_momentum_offloaded")
            if offloaded["active_bytes.current"] - initial["active_bytes.current"] != 2 * expected_block:
                raise ValueError("Two live pinned master/momentum banks do not match the expected block budget")
        host_peak = max(item["allocated_bytes.peak"] for item in receipt["observations"])
        host_test_peak = host_peak - initial["allocated_bytes.current"]
        gpu_test_peak = max(item["gpu_peak_allocated_bytes"] for item in receipt["observations"])
        if host_test_peak >= TEST_ALLOCATION_LIMIT_BYTES or gpu_test_peak >= TEST_ALLOCATION_LIMIT_BYTES:
            raise ValueError("Measured host-pinned or GPU test allocations exceeded the 2 GiB bound")
        receipt["allocation_bounds"] = {
            "limit_bytes": TEST_ALLOCATION_LIMIT_BYTES,
            "native_host_owned_peak_above_initial_bytes": host_test_peak,
            "native_gpu_tensor_peak_bytes": gpu_test_peak,
            "process_peak_rss_bytes": max(item["process_peak_rss_bytes"] for item in receipt["observations"]),
            "scope": "Tensor/allocator bound; imported process RSS reported separately under the 64 GiB pod",
        }
        receipt.update(
            result="passed",
            completed_storage_cycles=2,
            live_momentum_pointer_preserved=True,
            h2d_d2h_values_preserved=True,
            startup_active_bytes=startup["active_bytes.current"],
            final_two_bank_active_bytes=offloaded["active_bytes.current"],
        )
        write_receipt(args.output, receipt)
    except Exception as error:
        receipt.update(result="failed", error_type=type(error).__name__, error=str(error)[:1000])
        write_receipt(args.output, receipt)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=(*CASES, "all"), required=True)
    parser.add_argument("--momentum-timing", choices=TIMINGS, default="first_step")
    parser.add_argument("--optimizer-source", type=Path, required=True)
    parser.add_argument("--optimizer-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.plan:
        print(
            json.dumps(
                {
                    "cases": CASES,
                    "momentum_timings": TIMINGS,
                    "raw_bytes": RAW_BYTES,
                    "default_block_bytes": DEFAULT_BLOCK_BYTES,
                    "threshold_block_bytes": RAW_BYTES,
                    "gpu_allocation": "none; run requires separate reviewed submission",
                },
                indent=2,
            )
        )
        return
    if args.case != "all":
        run_case(args)
        return
    args.output.mkdir(parents=True)
    for case in CASES:
        for timing in TIMINGS:
            environment = dict(os.environ)
            environment.pop("PYTORCH_ALLOC_CONF", None)
            environment["PYTORCH_CUDA_ALLOC_CONF"] = ALLOCATOR_BASE + (
                ",pinned_max_round_threshold_mb:1" if case == "threshold" else ""
            )
            output = args.output / f"{case}-{timing}.json"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--case",
                    case,
                    "--momentum-timing",
                    timing,
                    "--optimizer-source",
                    str(args.optimizer_source.resolve()),
                    "--optimizer-sha256",
                    args.optimizer_sha256,
                    "--output",
                    str(output.resolve()),
                ],
                env=environment,
                check=True,
                timeout=180,
            )
    print(json.dumps({"result": "passed", "fresh_child_cases": 6, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
