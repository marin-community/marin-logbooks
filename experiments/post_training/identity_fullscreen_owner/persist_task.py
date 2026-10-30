# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""External task wrapper: frozen generation/math command, then finally-persist."""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.identity_fullscreen_owner.candidate import load_candidate, validate_candidate
from experiments.post_training.identity_fullscreen_owner.persistence import native_execution_id, persist_tree


def execute(candidate, kind, trial: str | None = None):
    validate_candidate(candidate)
    configure_coreweave_s3()
    execution_id = native_execution_id()
    base = Path(os.environ["IRIS_WORKDIR"])
    if kind == "generation":
        argv = candidate["screen"]["generation_argv"]
        output = base / "_pathb_fullscreen/outputs" / candidate["label"]
        remote = candidate["artifact_root"]
        unit = "generation-execution"
    else:
        if trial is None:
            raise ValueError("Math execution requires a bound trial")
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
    persistence_error = None
    try:
        command = [sys.executable, *argv[1:]]
        with (output / "task.log").open("wb") as log:
            completed = subprocess.run(command, cwd=base, stdout=log, stderr=subprocess.STDOUT, check=False)
        result["returncode"] = completed.returncode
        if completed.returncode:
            error = subprocess.CalledProcessError(completed.returncode, command)
    except BaseException as failure:
        result["error_type"] = type(failure).__name__
        error = failure
    finally:
        result["finished_unix"] = time.time()
        (output / "task-result.json").write_text(json.dumps(result, indent=2) + "\n")
        # Failure here remains a task failure; local files are retained, never
        # reported as uploaded. No replacement/retry or target repair occurs.
        try:
            persist_tree(
                output, remote, unit, execution_id, mirror_prefixes=("identity",) if kind == "generation" else ()
            )
        except Exception as failure:
            persistence_error = failure
            result["persistence_error_type"] = type(failure).__name__
            (output / "task-result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"Task persistence failed: {type(failure).__name__}: {failure}", file=sys.stderr)
    if error is not None:
        raise error from persistence_error
    if persistence_error is not None:
        raise persistence_error
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--smoke-persistence", action="store_true")
    parser.add_argument("--kind", required=True, choices=("generation", "math"))
    parser.add_argument("--trial", choices=("greedy", "17", "29"))
    args = parser.parse_args()
    candidate = load_candidate(args.candidate, args.candidate_sha256)
    if args.inspect_only:
        spec = (
            candidate["screen"]
            if args.kind == "generation"
            else next(item for item in candidate["math"] if item["trial"] == args.trial)
        )
        argv = spec["generation_argv"] if args.kind == "generation" else spec["argv"]
        smoke = (
            importlib.import_module(
                "experiments.post_training.identity_fullscreen_owner.owner_smoke"
            ).verify_persistence()
            if args.smoke_persistence
            else None
        )
        print(
            json.dumps(
                {
                    "persistence_smoke": smoke,
                    "candidate_path": str(args.candidate),
                    "candidate_sha256": args.candidate_sha256,
                    "label": candidate["label"],
                    "model_path": candidate["model_path"],
                    "artifact_root": candidate["artifact_root"],
                    "command_argv": argv,
                    "kind": args.kind,
                    "trial": args.trial,
                    "inspection_only": True,
                },
                sort_keys=True,
            )
        )
        return
    if args.smoke_persistence:
        raise ValueError("Dummy persistence smoke requires inspection-only execution")
    execute(candidate, args.kind, args.trial)


if __name__ == "__main__":
    main()
