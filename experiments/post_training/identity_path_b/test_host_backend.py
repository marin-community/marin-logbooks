# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only backend-admission regression; CUDA initialization is simulated."""

import json
import os
import subprocess
import sys

import pytest

# A fresh subprocess admits backends from JAX_PLATFORMS before any JAX backend
# initialization. CUDA visibility is simulated true (portable to CPU-only CI), and the CUDA
# initialization seam creates a real CPU client instead. This never initializes the resident local GPU and
# does not establish CUDA-device coverage. The actual CPU backend, JAX backend
# admission and native _single_cpu_sharding implementation are not mocked.
CPU_ONLY_PROBE = """
import json
from jax._src import xla_bridge

native_init = xla_bridge._init_backend
# CPU-only CI normally skips CUDA before the init seam. Simulate visibility;
# initialization below still creates CPU clients only, never a NVIDIA client.
xla_bridge.hardware_utils.has_visible_nvidia_gpu = lambda: True
initialized = []
def cpu_only_cuda_seam(platform):
    initialized.append(platform)
    return native_init("cpu" if platform == "cuda" else platform)
xla_bridge._init_backend = cpu_only_cuda_seam

from levanter.data.text.datasets import _single_cpu_sharding
try:
    sharding = _single_cpu_sharding()
except RuntimeError as error:
    print(json.dumps({"error": str(error), "initialized": initialized}))
else:
    print(json.dumps({"devices": [d.platform for d in sharding.device_set], "initialized": initialized}))
"""


@pytest.mark.parametrize("platforms,passes", [("cuda", False), ("cuda,cpu", True)])
def test_native_reader_cpu_backend_admitted_before_jax_initialization(platforms, passes):
    result = subprocess.run(
        [sys.executable, "-c", CPU_ONLY_PROBE],
        env={**os.environ, "JAX_PLATFORMS": platforms},
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    record = json.loads(result.stdout.splitlines()[-1])
    assert record["initialized"] == platforms.split(",")
    if passes:
        assert record["devices"] == ["cpu"]
    else:
        assert "Unknown backend cpu" in record["error"]
