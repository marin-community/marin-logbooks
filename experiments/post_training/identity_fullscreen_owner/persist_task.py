# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""External task wrapper: frozen generation/math command, then finally-persist."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.identity_fullscreen_owner.persistence import persist_tree


def execute(candidate, kind, trial=None):
    configure_coreweave_s3()
    base = Path(os.environ["IRIS_WORKDIR"])
    if kind == "generation":
        argv = candidate["screen"]["generation_argv"]
        output = base / "_pathb_fullscreen/outputs" / candidate["label"]
        remote = candidate["artifact_root"]
        unit = "generation-execution"
    else:
        spec = next(row for row in candidate["math"] if row["trial"] == trial)
        argv = spec["argv"]
        output = base / "_pathb_fullscreen/task-logs" / trial
        remote = f"{candidate['artifact_root']}/math/{trial}/execution"
        unit = "task-execution"
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "kind": kind,
        "trial": trial,
        "started_unix": time.time(),
        "command_argv": argv,
    }
    error = None
    try:
        command = [sys.executable, *argv[1:]]
        with (output / "task.log").open("wb") as log:
            completed = subprocess.run(command, cwd=base, stdout=log, stderr=subprocess.STDOUT, check=False)
        result["returncode"] = completed.returncode
    except BaseException as failure:
        result["error_type"] = type(failure).__name__
        error = failure
    finally:
        result["finished_unix"] = time.time()
        (output / "task-result.json").write_text(json.dumps(result, indent=2) + "\n")
        # Failure here remains a task failure; local files are retained, never
        # reported as uploaded. No replacement/retry or target repair occurs.
        persist_tree(output, remote, unit)
    if error is not None:
        raise error
    if result["returncode"]:
        raise RuntimeError("Frozen scientific command failed; available files were persisted")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=("generation", "math"))
    parser.add_argument("--trial", choices=("greedy", "17", "29"))
    args = parser.parse_args()
    execute(json.loads(args.candidate.read_text()), args.kind, args.trial)


if __name__ == "__main__":
    main()
