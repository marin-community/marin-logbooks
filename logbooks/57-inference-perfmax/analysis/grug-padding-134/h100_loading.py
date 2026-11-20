# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the existing FP8 loading regression against the installed candidate wheel."""

import hashlib
import importlib
import importlib.metadata
import json
import platform
import runpy
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import torch
import vllm


def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    root, output = Path(payload["source_root"]), Path(payload["output"])
    assert platform.machine() == "x86_64"
    assert torch.__version__ == "2.14.1+cu132"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.get_device_capability() == (9, 0)
    assert "H100" in torch.cuda.get_device_name()
    assert sys.flags.optimize == 0
    assert importlib.metadata.version("vllm") == payload["candidate_manifest"]["version"]
    package = Path(vllm.__file__).resolve().parent
    assert not package.is_relative_to(root.resolve())
    sources = {}
    for name, expected in (
        ("vllm.model_executor.models.grugmoe", payload["model_sha256"]),
        ("vllm.model_executor.layers.fused_moe.experts.marlin_moe", payload["marlin_sha256"]),
        ("vllm.model_executor.layers.fused_moe.layer", payload["factory_sha256"]),
    ):
        path = Path(importlib.import_module(name).__file__)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected
        sources[name] = {"path": str(path), "sha256": digest}
    native = {}
    for name, expected in payload["native_members"].items():
        path = package.parent / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert digest == expected
        native[name] = digest
    test = root / "tests/models/test_grugmoe.py"
    helper_path = root / "infra/release/gpu_validation.py"
    assert hashlib.sha256(test.read_bytes()).hexdigest() == payload["test_sha256"]
    assert hashlib.sha256(helper_path.read_bytes()).hexdigest() == payload["validation_helper_sha256"]
    record = {
        "status": "running",
        "source_commit": payload["source_commit"],
        "compiled_base": payload["compiled_base"],
        "research_source": payload["research_source"],
        "wheel_sha256": payload["candidate_manifest"]["wheels"][0]["sha256"],
        "package_path": str(package),
        "source_modules": sources,
        "native_payload_sha256": native,
        "test_sha256": payload["test_sha256"],
        "validation_helper_sha256": payload["validation_helper_sha256"],
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dependency_versions": {
            distribution.metadata["Name"]: distribution.version for distribution in importlib.metadata.distributions()
        },
        "started_unix": time.time(),
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
    sys.path.insert(0, str(helper_path.parent))
    helper = runpy.run_path(str(helper_path))
    arguments = [
        helper["source_node_id"](root, payload["pytest_node"]),
        "-vv",
        "-s",
        "--junitxml=" + str(output.with_suffix(".xml")),
    ]
    result = helper["run_wheel_tests"](
        SimpleNamespace(
            package_source_root=root,
            validation_source_root=root,
            exclude_test=[],
            pytest_args=arguments,
        )
    )
    cases = list(ET.parse(output.with_suffix(".xml")).getroot().iter("testcase"))
    counts = {
        "tests": len(cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
        "failures": sum(case.find("failure") is not None for case in cases),
        "errors": sum(case.find("error") is not None for case in cases),
    }
    passed = result == 0 and counts == {"tests": 1, "skipped": 0, "failures": 0, "errors": 0}
    record |= {
        "status": "passed" if passed else "failed",
        "pytest_exit": int(result),
        "pytest_args": arguments,
        "test_counts": counts,
        "test_names": [case.attrib.get("name") for case in cases],
        "finished_unix": time.time(),
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
