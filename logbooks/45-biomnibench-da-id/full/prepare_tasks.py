# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize a complete public-50 corpus from verified image manifests."""

import argparse
import hashlib
import json
import shlex
import shutil
from pathlib import Path, PurePosixPath

import toml

ROOT = Path(__file__).resolve().parent
DERIVED = {"da-1-3", "da-1-4", "da-17-1", "da-17-3", "da-17-5"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    args = parser.parse_args()
    groups = json.loads((args.plan_dir / "planned-images.json").read_text())
    images = {x["task"]: x for x in json.loads(args.images.read_text())}
    assert len(images) == 44
    manifest = []
    for group in groups:
        image = images[group["tasks"][0]]["image"]
        verified_inputs = images[group["tasks"][0]]["inputs"]
        assert len(verified_inputs) == len(group["inputs"])
        for original, verified in zip(group["inputs"], verified_inputs, strict=True):
            assert original["path"] == verified["path"] and original["size"] == verified["size"]
            assert original["sha256"] is None or original["sha256"] == verified["sha256"]
            assert len(verified["sha256"]) == 64
        assert "@sha256:" in image
        for task in group["tasks"]:
            source = args.source_dir / task
            inputs = verified_inputs
            # Image layers preserve the native Docker COPY destinations, including JSON COPY.
            for line in (source / "environment/Dockerfile").read_text().splitlines():
                if not line.startswith("COPY "):
                    continue
                pair = json.loads(line[5:]) if line[5:].startswith("[") else shlex.split(line[5:])
                assert len(pair) == 2, (task, pair)
                src, dest = pair
                expected = "/app/" + src
                actual = dest + PurePosixPath(src).name if dest.endswith("/") and not src.endswith("/") else dest
                assert expected == actual, (task, pair)
            original = (source / "instruction.md").read_text()
            instruction = original
            if task in DERIVED:
                exemplar = "da-1-3" if task.startswith("da-1-") else "da-17-3"
                prior = (ROOT.parent / "pilot/prepared-tasks" / exemplar / "instruction.md").read_text()
                disclosure = prior.split("## Input variant disclosure\n\n")[1].split("## Data Files")[0]
                instruction = instruction.replace(
                    "## Data Files", "## Input variant disclosure\n\n" + disclosure + "## Data Files", 1
                )
            config = toml.loads((source / "task.toml").read_text())
            native = dict(config["environment"])
            config["environment"].update(
                docker_image=image,
                workdir="/app",
                memory_mb=min(native["memory_mb"], 8192),
                storage_mb=min(native["storage_mb"], 10240),
            )
            config["verifier"]["env"] = {}
            config["verifier"]["timeout_sec"] = 900
            config["agent"]["timeout_sec"] = 3600
            for directory in ("runtime-tasks", "tasks"):
                target = ROOT / directory / task
                (target / "tests").mkdir(parents=True, exist_ok=True)
                (target / "environment").mkdir(exist_ok=True)
                (target / "instruction.md").write_text(instruction)
                shutil.copyfile(source / "tests/rubric.txt", target / "tests/rubric.txt")
                shutil.copyfile(ROOT.parent / "pilot/llm_judge.py", target / "tests/llm_judge.py")
                shutil.copyfile(ROOT / "grade_with_retry.py", target / "tests/grade_with_retry.py")
                shutil.copyfile(
                    ROOT.parent / "pilot/remaining-tasks/da-20-3/tests/judge-config.json",
                    target / "tests/judge-config.json",
                )
                (target / "tests/test.sh").write_text(
                    "#!/bin/bash\nset -euo pipefail\nrm -f /logs/verifier/reward.json /logs/verifier/re"
                    "ward.txt\nexec /opt/biomnibench-judge/bin/python /tests/grade_with_retry.py\n"
                )
                effective = (
                    image
                    if directory == "runtime-tasks"
                    else "registry.example.invalid/biomnibench-da@" + image.split("@")[1]
                )
                config["environment"]["docker_image"] = effective
                (target / "task.toml").write_text(toml.dumps(config))
                (target / "environment/shellbox-images.json").write_text(
                    json.dumps({"main": {"image": effective, "workdir": "/app"}}, indent=2) + "\n"
                )
            manifest.append(
                {
                    "task": task,
                    "input_variant": task in DERIVED,
                    "image_digest": image.split("@")[1],
                    "input_files": inputs,
                    "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
                    "rubric_sha256": hashlib.sha256((source / "tests/rubric.txt").read_bytes()).hexdigest(),
                    "resources": {k: config["environment"][k] for k in ("cpus", "memory_mb", "storage_mb", "gpus")},
                }
            )
    assert len(manifest) == 50 and sum(x["input_variant"] for x in manifest) == 5
    (ROOT / "task-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"tasks": len(manifest), "variants": 5, "status": "prepared; runtime checks pending"}))


if __name__ == "__main__":
    main()
