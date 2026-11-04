# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit one published grouped-Gram acceleration diagnostic with default Iris retries."""

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def launch_argv(source_commit):
    directory = "logbooks/20-ema-gram-polar"
    script = f"{directory}/grouped_gram_check.py"
    manifest = f"{directory}/grouped-gram-manifest.json"
    prefix = "gs://marin-us-central2/ema-gram-20-grouped-gram/2026.11.04.01"
    checkpoint = json.loads((ROOT / directory / "loss-pair-checkpoint-metadata.json").read_text())["checkpoint"]
    common = ["python", script, "--source-manifest", manifest, "--samples", "3"]
    logging = [*common, "--mode", "logging-check", "--output-prefix", prefix + "/host-logging"]
    worker = [*common, "--mode", "worker", "--checkpoint", checkpoint, "--output-prefix", prefix]
    command = "JAX_PLATFORMS=cpu " + shlex.join(logging) + " && exec " + shlex.join(worker)
    return [
        "uv",
        "run",
        "--no-sync",
        "iris",
        "--cluster=marin",
        "job",
        "run",
        "--no-wait",
        "--no-cancel-on-exit",
        "--job-name",
        "ema-gram-20-grouped-gram-v4-32-20261104",
        "--priority",
        "batch",
        "--enable-extra-resources",
        "--tpu",
        "v4-32",
        "--cpu",
        "120",
        "--memory",
        "256GB",
        "--disk",
        "50GB",
        "--region",
        "us-central2",
        "--extra",
        "tpu",
        "--sync-package",
        "marin-core",
        "--sync-package",
        "marin-levanter",
        "--sync-package",
        "marin-dupekit",
        "-e",
        "GIT_COMMIT",
        source_commit,
        "--",
        "bash",
        "-c",
        command,
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.source_commit
    base = Path(__file__).parent
    manifest = json.loads((base / "grouped-gram-manifest.json").read_text())
    for path, identity in manifest["files"].items():
        raw = (ROOT / path).read_bytes()
        assert len(raw) == identity["bytes"] and hashlib.sha256(raw).hexdigest() == identity["sha256"]
    for file in ["grouped-gram-logging-validation.json", "grouped-gram-wandb-validation.json"]:
        evidence = json.loads((base / file).read_text())
        assert evidence["state"] == "finished" and evidence["source_files"] == manifest["files"]
    cpu = json.loads((base / "grouped-gram-cpu-validation.json").read_text())
    assert cpu["completed"] and cpu["source_files"] == manifest["files"]
    for mode in ["single", "matrix"]:
        assert json.loads((base / f"grouped-gram-regression-{mode}-validation.json").read_text())["exit_code"] == 0
    argv = launch_argv(args.source_commit)
    print(json.dumps(argv), flush=True)
    if args.launch:
        subprocess.run(argv, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
