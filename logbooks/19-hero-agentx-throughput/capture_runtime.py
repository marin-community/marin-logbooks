# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture the installed serving environment without dumping credentials."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import torch

SOURCE_PATHS = (
    "vllm/model_executor/models/grugmoe.py",
    "vllm/model_executor/models/config.py",
    "vllm/transformers_utils/configs/grugmoe.py",
    "vllm/config/model.py",
    "vllm/config/cache.py",
    "vllm/v1/attention/backends/short_conv_attn.py",
    "vllm/v1/attention/backends/mamba_attn.py",
    "vllm/model_executor/layers/mamba/ops/causal_conv1d.py",
    "vllm/model_executor/model_loader/reload/layerwise.py",
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    distribution = importlib.metadata.distribution("vllm")
    root = Path(distribution.locate_file("")).resolve()
    extensions = sorted(
        str(path) for path in distribution.files or () if str(path).startswith("vllm/") and str(path).endswith(".so")
    )
    core = next(path for path in extensions if Path(path).name.startswith("_C"))
    imported = importlib.import_module("vllm." + Path(core).name.split(".")[0])
    assert Path(imported.__file__).resolve().is_relative_to(root)
    record = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "node": platform.node(),
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())),
        "vllm_direct_url": json.loads(distribution.read_text("direct_url.json")),
        "torch_import_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "nccl_version": torch.cuda.nccl.version(),
        "core_extension_import_path": imported.__file__,
        "source_sha256": {path: digest(root / path) for path in SOURCE_PATHS},
        "extension_sha256": {path: digest(root / path) for path in extensions},
        "gpus": (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader"], text=True
            ).splitlines()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Captured {len(record['packages'])} packages and {len(extensions)} extensions: {args.output}")


if __name__ == "__main__":
    main()
