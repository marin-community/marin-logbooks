# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze Marin's serving runtime or an explicit build without adopting it."""

import dataclasses
import hashlib
import json
import platform
import tempfile
from pathlib import Path
from urllib.parse import quote

import draccus
import requests
from marin.external_dependencies import CUDA_TOOLCHAIN_VERSION_BY_BACKEND, VLLM_GPU_RELEASE, VllmGpuRelease
from marin.inference.vllm_release import load_vllm_gpu_release

from experiments.evaluation.knotmeter.artifacts import copy_file, write_record
from experiments.evaluation.knotmeter.config import Runtime


def resolve_release(runtime: Runtime) -> dict:
    release = VLLM_GPU_RELEASE
    location = runtime.manifest
    if runtime.release:
        location = (
            f"https://github.com/marin-community/vllm/releases/download/{quote(runtime.release, safe='')}/"
            "marin-vllm-gpu-manifest.json"
        )
    if location:
        with tempfile.TemporaryDirectory(prefix="knotmeter-runtime-") as directory:
            path = Path(directory) / "manifest.json"
            if location.startswith(("https://", "http://")):
                response = requests.get(location, timeout=60)
                response.raise_for_status()
                path.write_bytes(response.content)
            else:
                copy_file(location, str(path))
            release = load_vllm_gpu_release(path)
    if release.torch_backend not in CUDA_TOOLCHAIN_VERSION_BY_BACKEND:
        raise ValueError(f"No registered CUDA toolchain for {release.torch_backend!r}")
    if runtime.release and release.release_tag != runtime.release:
        raise ValueError("Runtime manifest does not match the requested release")
    if runtime.revision and release.source_commit != runtime.revision:
        raise ValueError("Runtime manifest does not match the requested source revision")
    return json.loads(json.dumps(dataclasses.asdict(release)))


def gpu_release(metadata: dict) -> VllmGpuRelease:
    return draccus.decode(VllmGpuRelease, metadata)


def materialize_release(metadata: dict, output: Path) -> VllmGpuRelease:
    """Install a private experimental wheel from its durable cloud URI."""
    release = gpu_release(metadata)
    wheels = []
    for wheel in release.wheels:
        if wheel.architecture != platform.machine() or not wheel.url.startswith(("s3://", "gs://")):
            wheels.append(wheel)
            continue
        directory = Path(tempfile.gettempdir()) / "knotmeter-wheels" / wheel.sha256
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / wheel.url.rsplit("/", 1)[-1]
        if not path.is_file():
            copy_file(wheel.url, str(path))
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != wheel.sha256:
            raise ValueError("Experimental wheel checksum differs from its immutable manifest")
        write_record(
            str(output / "runtime-wheel.json"),
            {"uri": wheel.url, "sha256": actual, "local_uri": path.as_uri(), "source_commit": release.source_commit},
        )
        wheels.append(dataclasses.replace(wheel, url=path.as_uri()))
    return dataclasses.replace(release, wheels=tuple(wheels))
