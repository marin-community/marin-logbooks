# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize only the three new tasks after successful image validation."""

import json
import shutil
from pathlib import Path

import toml
import yaml

root = Path(__file__).resolve().parent
images = json.loads((root / "remaining-images.json").read_text())
reports = json.loads((root / "remaining-runtime-report.json").read_text())
assert len(images) == len(reports) == 3
assert all(r["status"] == "passed" and r["cleanup"] == "closed" for r in reports)
for image in images:
    task = image["task"]
    assert any(r["task"] == task and r["image"] == image["image"] for r in reports)
    source = root / "prepared-tasks" / task
    destination = root / "remaining-tasks" / task
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "instruction.md", destination / "instruction.md")
    shutil.copytree(source / "tests", destination / "tests", dirs_exist_ok=True)
    config = toml.loads((root / "source-metadata" / task / "task.toml").read_text())
    config["environment"].update(docker_image=image["image"], workdir="/app", memory_mb=8192, storage_mb=10240)
    config["verifier"]["env"] = {}
    (destination / "task.toml").write_text(toml.dumps(config))
    (destination / "environment").mkdir(exist_ok=True)
    (destination / "environment" / "shellbox-images.json").write_text(
        json.dumps({"main": {"image": image["image"], "workdir": "/app"}}, indent=2) + "\n"
    )
config = yaml.safe_load((root / "harbor.yaml").read_text())
config["datasets"] = [{"path": "remaining-tasks"}]
(root / "remaining.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
print("Prepared exactly three remaining tasks; original control preserved.")
