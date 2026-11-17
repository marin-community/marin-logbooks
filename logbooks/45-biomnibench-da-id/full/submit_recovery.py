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
    parser.add_argument("--job-name", required=True)
    args = parser.parse_args()
    receipt = ROOT / "recovery-launcher-submission.json"
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
                "logbooks/45-biomnibench-da-id/full/launch_recovery.py",
                "--submit-in-cluster",
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
            timeout=Duration.from_seconds(28800),
            scheduling_timeout=Duration.from_seconds(1800),
            max_retries_failure=0,
            max_retries_preemption=0,
        )
        result = {"job_id": str(job.job_id), "source_commit": source, "parent_timeout_seconds": 28800}
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
