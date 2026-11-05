# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise real Iris ConfigMap/blob staging for the new flat allocator payload."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_flat_allocator_payload_runs_real_worker_preflight(tmp_path, monkeypatch):
    root = Path(__file__).parent
    submit = module(root / "submit.py", "host_allocator_submit")
    previous = module(root.parent / "hero_shortconv_validation/test_staging.py", "closed_staging_regression")
    workspace, payload, manifest = submit.prepare(tmp_path / "prepared")
    assert all(Path(name).parent == Path(".") for name in payload)
    workdir = tmp_path / "flat-workdir"
    previous._project_and_stage(payload, tmp_path / "projection", workdir, monkeypatch)
    for name, content in payload.items():
        assert (workdir / name).read_bytes() == content
    for path in manifest["managed_inputs"]:
        target = workdir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workspace / path, target)
    result = subprocess.run(
        [
            sys.executable,
            str(workdir / submit.WORKER_FILENAME),
            "--manifest",
            str(workdir / submit.MANIFEST_FILENAME),
            "--manifest-sha256",
            submit.digest(payload[submit.MANIFEST_FILENAME]),
            "--check-staged",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    receipt = json.loads(result.stdout)
    assert receipt["result"] == "staged-payload-valid"
    assert receipt["workspace"] == str(workdir)
    planned = subprocess.run(
        [
            sys.executable,
            str(workdir / submit.PROBE_FILENAME),
            "--case",
            "all",
            "--optimizer-source",
            str(workdir / submit.OPTIMIZER_FILENAME),
            "--optimizer-sha256",
            manifest["optimizer_sha256"],
            "--output",
            str(tmp_path / "not-executed"),
            "--plan",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    plan = json.loads(planned.stdout)
    assert plan["raw_bytes"] == 36 * 1024**2
    assert plan["default_block_bytes"] == 64 * 1024**2
    assert plan["threshold_block_bytes"] == 36 * 1024**2
    assert not (tmp_path / "not-executed").exists()
