# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package the Python-only candidate with native, verified precompiled payloads."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

import torch


def native_members(path: Path) -> dict[str, str]:
    members = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                if stream.read(4) == b"\x7fELF":
                    with archive.open(name) as native:
                        members[name] = hashlib.file_digest(native, "sha256").hexdigest()
    return members


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text())
    root, base = Path(payload["source_root"]), Path(payload["base_wheel"])
    architecture = payload.get("architecture", "aarch64")
    assert architecture in {"aarch64", "x86_64"}, architecture
    assert torch.__version__ == "2.14.1+cu132", torch.__version__
    source = root / "vllm/model_executor/models/grugmoe.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == payload["model_sha256"]
    marlin = root / "vllm/model_executor/layers/fused_moe/experts/marlin_moe.py"
    assert hashlib.sha256(marlin.read_bytes()).hexdigest() == payload["marlin_sha256"]
    factory = root / "vllm/model_executor/layers/fused_moe/layer.py"
    assert hashlib.sha256(factory.read_bytes()).hexdigest() == payload["factory_sha256"]
    environment = os.environ | {
        "VLLM_USE_PRECOMPILED": "1",
        "VLLM_PRECOMPILED_WHEEL_LOCATION": str(base),
        "VLLM_TARGET_DEVICE": "cuda",
        "VLLM_VERSION_OVERRIDE": payload["version"],
        "VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX": "1",
        "SOURCE_DATE_EPOCH": str(payload["source_date_epoch"]),
    }
    # The supplied base wheel contains the target binaries. When Iris places
    # the CPU job on another architecture, setuptools only packages those bytes.
    command = [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(root / "dist")]
    if platform.machine() != architecture:
        command += [f"--config-setting=--build-option=--plat-name=linux_{architecture}"]
    command += [str(root)]
    subprocess.run(command, cwd=root, env=environment, check=True)
    wheels = list((root / "dist").glob("*.whl"))
    assert len(wheels) == 1, wheels
    wheel = wheels[0]
    assert wheel.name.endswith(f"linux_{architecture}.whl"), wheel.name
    baseline_native, candidate_native = native_members(base), native_members(wheel)
    assert candidate_native and candidate_native == baseline_native, {
        "missing": sorted(baseline_native.keys() - candidate_native.keys()),
        "added": sorted(candidate_native.keys() - baseline_native.keys()),
        "changed": [
            name
            for name in candidate_native
            if name in baseline_native and candidate_native[name] != baseline_native[name]
        ],
    }
    with zipfile.ZipFile(wheel) as archive:
        core_name = next(name for name in candidate_native if name.startswith("vllm/_moe_C") and name.endswith(".so"))
        with archive.open(core_name) as core:
            elf_architecture = int.from_bytes(core.read(20)[18:20], "little")
        assert elf_architecture == {"aarch64": 183, "x86_64": 62}[architecture]
        for name, expected in payload.get("python_hashes", {}).items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
        model_sha = hashlib.sha256(archive.read("vllm/model_executor/models/grugmoe.py")).hexdigest()
        marlin_sha = hashlib.sha256(
            archive.read("vllm/model_executor/layers/fused_moe/experts/marlin_moe.py")
        ).hexdigest()
        factory_sha = hashlib.sha256(archive.read("vllm/model_executor/layers/fused_moe/layer.py")).hexdigest()
    assert model_sha == payload["model_sha256"]
    assert marlin_sha == payload["marlin_sha256"]
    assert factory_sha == payload["factory_sha256"]
    with wheel.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    record = {
        "scope": "Experimental Python-only wheel; native setup.py precompiled build, no source overlay",
        "source_commit": payload["source_commit"],
        "compiled_base": payload["compiled_base"],
        "base_wheel_sha256": payload["base_wheel_sha256"],
        "version": payload["version"],
        "wheel_path": str(wheel),
        "wheel_sha256": digest,
        "wheel_bytes": wheel.stat().st_size,
        "model_sha256": model_sha,
        "marlin_sha256": marlin_sha,
        "factory_sha256": factory_sha,
        "native_members": candidate_native,
        "native_payloads_equal_base": True,
        "command": command,
        "build_environment": {
            key: environment[key] for key in environment if key.startswith("VLLM_") or key == "SOURCE_DATE_EPOCH"
        },
        "python": sys.version,
        "machine": platform.machine(),
        "architecture": architecture,
        "torch": torch.__version__,
    }
    Path(payload["output"]).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
