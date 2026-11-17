# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze Marin's serving runtime or an explicit build without adopting it."""

import dataclasses
import json
import tempfile
from pathlib import Path
from urllib.parse import quote

import draccus
import requests
from marin.external_dependencies import VLLM_GPU_RELEASE, VllmGpuRelease
from marin.inference.vllm_release import load_vllm_gpu_release

from experiments.evaluation.knotmeter.artifacts import copy_file
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
        assert release is not None
    if runtime.release and release.release_tag != runtime.release:
        raise ValueError("Runtime manifest does not match the requested release")
    if runtime.revision and release.source_commit != runtime.revision:
        raise ValueError("Runtime manifest does not match the requested source revision")
    return json.loads(json.dumps(dataclasses.asdict(release)))


def gpu_release(metadata: dict) -> VllmGpuRelease:
    return draccus.decode(VllmGpuRelease, metadata)
