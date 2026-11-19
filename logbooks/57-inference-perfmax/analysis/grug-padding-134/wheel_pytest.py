# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the fork's existing wheel-test helper in a fresh process."""

import hashlib
import importlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    for name, expected in (
        ("vllm.model_executor.models.grugmoe", payload["model_sha256"]),
        ("vllm.model_executor.layers.fused_moe.experts.marlin_moe", payload["marlin_sha256"]),
    ):
        module = importlib.import_module(name)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected
    root = Path(payload["source_root"])
    output = Path(payload["output"])
    sys.path.insert(0, str(root / "infra/release"))
    helper = runpy.run_path(str(root / "infra/release/gpu_validation.py"))
    result = helper["run_wheel_tests"](
        SimpleNamespace(
            package_source_root=root,
            validation_source_root=root,
            exclude_test=[],
            pytest_args=[
                str(root / "tests/models/test_grugmoe.py"),
                "-k",
                payload["pytest_selection"],
                "-v",
                "--junitxml=" + str(output.with_suffix(".xml")),
            ],
        )
    )
    output.write_text(
        json.dumps({"source_commit": payload["source_commit"], "pytest_exit": result, "fresh_process": True}, indent=2)
        + "\n"
    )
    raise SystemExit(result)


if __name__ == "__main__":
    main()
