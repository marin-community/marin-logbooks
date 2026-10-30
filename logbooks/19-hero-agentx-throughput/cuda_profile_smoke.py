# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check native CUDA-profiler API capture across four spawned GPU workers.

Run through the recorded Nsight prefix only after throughput windows finish.
This synthetic collector check makes no Hero correctness or throughput claim.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.multiprocessing as multiprocessing


def worker(rank: int, barrier, output: str) -> None:
    torch.cuda.set_device(rank)
    values = torch.ones((64, 64), dtype=torch.float32, device=f"cuda:{rank}")
    result = values @ values
    torch.cuda.synchronize()
    barrier.wait(timeout=120)
    torch.cuda.profiler.start()
    with torch.cuda.nvtx.range(f"collector-smoke-device-{rank}"):
        for _ in range(3):
            result = values @ values
    torch.cuda.synchronize()
    barrier.wait(timeout=120)
    torch.cuda.profiler.stop()
    assert bool(torch.all(result == 64).item())
    record = {
        "device": rank,
        "name": torch.cuda.get_device_name(rank),
        "pid": os.getpid(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "known_matmul_result": 64,
    }
    (Path(output) / f"device-{rank}.json").write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.__version__ == "2.14.1+cu132" and torch.version.cuda == "13.2"
    assert torch.cuda.device_count() == 4
    args.output.mkdir(parents=True, exist_ok=False)
    barrier = multiprocessing.get_context("spawn").Barrier(4)
    multiprocessing.start_processes(worker, args=(barrier, str(args.output)), nprocs=4, start_method="spawn")
    assert len(list(args.output.glob("device-*.json"))) == 4
    print("CUDA_PROFILE_FOUR_WORKER_SMOKE_PASSED", flush=True)


if __name__ == "__main__":
    main()
