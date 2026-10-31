# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile installed native CUTLASS under the pinned isolated CUDA interpreter."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from flashinfer.jit import env as jit_env
from flashinfer.jit.core import current_compilation_context
from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm100_module


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def worker(phase: str, output: Path) -> None:

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and torch.cuda.device_count() == 0
    assert not torch.cuda.is_initialized()
    assert importlib.metadata.version("flashinfer-python") == "0.6.18.post1"
    assert importlib.metadata.version("torch") == "2.14.1+cu132"
    assert current_compilation_context.TARGET_CUDA_ARCHS == {(10, "0a")}
    spec = gen_cutlass_fused_moe_sm100_module()
    spec.write_ninja()
    record = {
        "phase": phase,
        "started_unix": time.time(),
        "versions": {
            name: importlib.metadata.version(name) for name in ["vllm", "torch", "flashinfer-python", "nvidia-cuda-nvcc"]
        },
        "ninja_path": str(spec.ninja_path),
        "ninja_sha256": hashlib.sha256(spec.ninja_path.read_bytes()).hexdigest(),
        "library_path": str(spec.jit_library_path),
        "workspace": str(jit_env.FLASHINFER_WORKSPACE_DIR),
        "target_cuda_archs": sorted(current_compilation_context.TARGET_CUDA_ARCHS),
        "nvcc_flags": current_compilation_context.get_nvcc_flags_list(supported_major_versions=[10, 11]),
        "cuda_initialized": False,
    }
    write(output / (phase + "-worker.json"), record)
    if phase == "preflight":
        command = ["ninja", "-n", "-C", str(spec.build_dir), "-f", str(spec.ninja_path), "-j", "8"]
        subprocess.run(command, check=True, timeout=60)
    else:
        spec.build_and_load()
        assert spec.jit_library_path.is_file()
        record.update(
            library_sha256=hashlib.sha256(spec.jit_library_path.read_bytes()).hexdigest(),
            library_bytes=spec.jit_library_path.stat().st_size,
        )
    assert not torch.cuda.is_initialized()
    record.update(status="passed", finished_unix=time.time())
    write(output / (phase + "-worker.json"), record)
    print(json.dumps(record), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "build"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    worker(args.phase, args.output)


if __name__ == "__main__":
    main()
