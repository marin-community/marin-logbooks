# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute the Daytona snapshot name each SWE-bench Verified task would create.

Uses the production naming path (Harbor's machine_spec and Shellbox's snapshot_identity),
mirroring experiments/post_training/task_curation/daytona_identity.py. No Daytona API calls.
"""

import json
import math
import sys
import tempfile
import tomllib
from pathlib import Path

from daytona import Resources
from harbor.environments.daytona import machine_spec
from harbor_config.models.task.config import EnvironmentConfig
from huggingface_hub import snapshot_download
from shellbox.backends.daytona.machine import snapshot_identity

OUT = Path(sys.argv[1])
root = Path(
    snapshot_download(
        repo_id="DCAgent/swebench-verified",
        repo_type="dataset",
        allow_patterns=["*/task.toml", "*/environment/*"],
    )
)
names: dict[str, str] = {}
failures: dict[str, str] = {}
for task_dir in sorted(p for p in root.iterdir() if (p / "task.toml").exists()):
    try:
        config = EnvironmentConfig(**tomllib.loads((task_dir / "task.toml").read_text()).get("environment", {}))
        # The Hub cache holds symlinks; stage real files with the mode an extracted task tree carries.
        context = Path(tempfile.mkdtemp())
        for source in (task_dir / "environment").iterdir():
            target = context / source.name
            target.write_bytes(source.read_bytes())
            target.chmod(0o644)
        spec = machine_spec(
            config, context, {}, cpus=config.cpus, memory_mb=config.memory_mb, storage_mb=config.storage_mb
        )
        resources = Resources(
            cpu=spec.cpus,
            memory=math.ceil(spec.memory_mb / 1024) if spec.memory_mb is not None else None,
            disk=math.ceil(spec.storage_mb / 1024) if spec.storage_mb is not None else None,
            gpu=spec.gpus or None,
        )
        names[task_dir.name] = snapshot_identity(spec.source, resources)
    except Exception as error:  # report per-task failures; the count must be complete
        failures[task_dir.name] = f"{type(error).__name__}: {error}"
OUT.write_text(json.dumps({"names": names, "failures": failures}, indent=1))
print("tasks", len(names), "distinct", len(set(names.values())), "failures", len(failures))
for task, reason in list(failures.items())[:3]:
    print("failure", task, reason[:200])
