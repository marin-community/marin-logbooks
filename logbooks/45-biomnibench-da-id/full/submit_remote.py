# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the validated public-50 launcher without loading evaluation dependencies locally."""

import argparse
import getpass
import json
import subprocess
from pathlib import Path

from iris.cli.connect import open_iris_client
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from rigging.secrets import require_secret, resolve_secret_spec
from rigging.timing import Duration

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    candidates = parser.add_mutually_exclusive_group()
    candidates.add_argument("--gemma", action="store_true")
    candidates.add_argument("--gpt-oss", action="store_true")
    parser.add_argument("--final-task", action="store_true")
    parser.add_argument("--two-day-window", action="store_true")
    args = parser.parse_args()
    if args.final_task and not (args.gpt_oss or args.gemma):
        parser.error("--final-task requires --gpt-oss or --gemma")
    cap = 7200 if args.final_task else 28800
    if args.two_day_window:
        if not (args.gemma and args.final_task):
            parser.error("--two-day-window applies to the final Gemma task")
        cap = 48 * 3600
    validation = json.loads(args.validation.read_text())
    assert {key: validation[key] for key in ("status", "snapshots", "runtime_checks", "all_closed")} == {
        "status": "passed",
        "snapshots": 45,
        "runtime_checks": 7,
        "all_closed": True,
    }
    receipt = ROOT / ("gpt-oss-launcher-submission.json" if args.gpt_oss else "launcher-submission.json")
    if args.gemma:
        receipt = ROOT / "gemma-launcher-submission.json"
    if args.final_task:
        receipt = ROOT / ("gemma-final-submission.json" if args.gemma else "gpt-oss-final-submission.json")
    if args.two_day_window:
        receipt = ROOT / "gemma-final-48h-submission.json"
    assert not receipt.exists(), "Submission already recorded; inspect before any retry"
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    secrets = {name: require_secret(name) for name in ("CW_KEY_ID", "CW_KEY_SECRET")}
    secrets["DAYTONA_API_KEY"] = resolve_secret_spec(
        "gcp-secret://projects/hai-gcp-models/secrets/DAYTONA_EVAL_API_KEY/versions/latest"
    ).value
    with open_iris_client(cluster_name="marin", workspace=ROOT.parents[2]) as client:
        job = client.submit(
            entrypoint=Entrypoint.from_command(
                "uv",
                "run",
                "--no-sync",
                "python",
                "logbooks/45-biomnibench-da-id/full/launch_full.py",
                "--submit-in-cluster",
                *(["--two-day-window"] if args.two_day_window else []),
                *(["--final-task"] if args.final_task else []),
                *(["--gemma"] if args.gemma else []),
                *(["--gpt-oss"] if args.gpt_oss else []),
            ),
            name=args.job_name,
            resources=ResourceSpec(cpu=1, memory="2g", disk="16g"),
            environment=EnvironmentSpec(
                env_vars={
                    "PYTHONPATH": ".",
                    "MARIN_GIT_SHA": source,
                    "MARIN_EVAL_USER": getpass.getuser(),
                    "DAYTONA_TARGET": "us",
                },
                secrets=secrets,
                sync_packages=["marin-core"],
            ),
            constraints=[
                Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value="cw-us-east-02a"),
                preemptible_constraint(False),
            ],
            timeout=Duration.from_seconds(cap),
            scheduling_timeout=Duration.from_seconds(cap if args.two_day_window else 1800),
            max_retries_failure=0,
            max_retries_preemption=0,
        )
        result = {"job_id": str(job.job_id), "source_commit": source, "parent_timeout_seconds": cap}
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
