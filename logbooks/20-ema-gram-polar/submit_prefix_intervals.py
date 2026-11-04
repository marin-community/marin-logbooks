# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit CPU profile parsing outside the currently measured training slice."""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from iris.cli.connect import open_iris_client
from iris.cluster.constraints import Constraint, ConstraintOp, region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", choices=["intervals", "assignment"], default="intervals")
    parser.add_argument("--job-name")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest_path = Path(f"logbooks/20-ema-gram-polar/prefix-{args.analysis}-manifest.json")
    raw = (root / manifest_path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.manifest_sha256
    manifest = json.loads(raw)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.source_commit
    for path, sha in manifest["source_sha256"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == sha
    command = [
        "python",
        "logbooks/20-ema-gram-polar/profile_prefix_intervals_regionally.py",
        "--analysis",
        args.analysis,
        "--manifest",
        str(manifest_path),
        "--manifest-sha256",
        args.manifest_sha256,
        "--output-prefix",
        f"gs://marin-us-central2/ema-gram-20-profile/2026.11.04.prefix-{args.analysis}",
    ]
    constraints = [
        region_constraint(["us-central2"]),
        Constraint.create(key="tpu-name", op=ConstraintOp.NE, value=manifest["excluded_training_tpu_name"]),
    ]
    with open_iris_client(config_file=root / "lib/iris/config/marin.yaml", workspace=root) as client:
        job = client.submit(
            entrypoint=Entrypoint.from_command(*command),
            name=args.job_name or f"ema-gram-20-prefix-{args.analysis}-cpu-20261104",
            resources=ResourceSpec(cpu=8, memory="64GB", disk="20GB"),
            environment=EnvironmentSpec(env_vars={"GIT_COMMIT": args.source_commit}, sync_packages=["marin-core"]),
            constraints=constraints,
            priority_band=job_pb2.PRIORITY_BAND_BATCH,
        )
        print(
            json.dumps(dict(job_id=str(job.job_id), accepted_utc=datetime.now(UTC).isoformat(), command=command)),
            flush=True,
        )


if __name__ == "__main__":
    main()
