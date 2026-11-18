# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run native component tests with a declared Python-module override."""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import vllm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text())
    root = Path(payload["source_root"])
    model = root / "vllm/model_executor/models/grugmoe.py"
    actual_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    if actual_hash != payload["model_sha256"]:
        raise RuntimeError("The component model source differs from the published input")
    if "0ac9dcb157ee" not in importlib.metadata.version("vllm"):
        raise RuntimeError("The compiled base is not the pinned serving wheel")
    if torch.__version__ != "2.14.1+cu132":
        raise RuntimeError("Torch differs from the pinned compiled base")
    spec = importlib.util.spec_from_file_location("vllm.model_executor.models.grugmoe", model)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    importlib.import_module("vllm.model_executor.models").grugmoe = module
    record = {
        "scope": "Component diagnostics; compiled base plus one explicit Python-module override",
        "compiled_base": "0ac9dcb157ee86ccb447bc717259159c41ad456a",
        "source_commit": payload["source_commit"],
        "vllm_package": vllm.__file__,
        "model_module": module.__file__,
        "model_sha256": actual_hash,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
    }
    Path(payload["output"]).write_text(json.dumps(record, indent=2) + "\n")
    sys.path.insert(0, str(root / "infra/release"))
    native = runpy.run_path(str(root / "infra/release/gpu_validation.py"))
    result = native["run_wheel_tests"](
        SimpleNamespace(
            package_source_root=root,
            validation_source_root=root,
            exclude_test=[],
            pytest_args=[
                str(root / "tests/models/test_grugmoe.py"),
                "-k",
                "online_fp8_loading_matches_vllm or hero_online_fp8_latent_and_shared "
                "or quantization_platform_support or preserves_gpu_tensor_parallel_restriction",
                "-v",
                "--maxfail=1",
                "--junitxml=" + str(Path(payload["output"]).with_suffix(".xml")),
            ],
        )
    )
    record["pytest_exit"] = result
    Path(payload["output"]).write_text(json.dumps(record, indent=2) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
