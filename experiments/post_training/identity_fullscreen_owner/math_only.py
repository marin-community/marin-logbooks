# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native math-only lifetime owner for committed source-bound candidates."""

import argparse
import hashlib
import shlex
from pathlib import Path

from fray.iris_backend import convert_constraints
from fray.types import ResourceConfig
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import SecretBehavior
from rigging.timing import Duration

from experiments.post_training.identity_fullscreen_owner.candidate import bundled_candidate, load_candidate
from experiments.post_training.identity_fullscreen_owner.owner import OWNER_SECONDS, run

ARCHIVE_PATH = "_pathb_math/f5-workspace.zip"
ARCHIVE_SHA256 = "2f0d5ce76bb408f0e803bd51044472af11ddb77b364443d9b09c1a5f5ab7379d"


def check_model(candidate: dict) -> None:
    if candidate["scope"] != "math-only":
        raise ValueError("This constructor only admits paired maths")
    for item in candidate["model_guard"]:
        raw = StoragePath(item["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("Math source provenance changed before serve submission")


def submission(*, name: str, candidate: dict, archive: bytes, credentials: dict[str, str]) -> dict:
    if candidate["scope"] != "math-only" or not name.startswith("pathb-paired-math-"):
        raise ValueError("Require the exact paired-maths candidate")
    if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Pinned f5 workspace changed")
    path, digest = bundled_candidate(candidate)
    secrets = {key: credentials[key] for key in ("CW_KEY_ID", "CW_KEY_SECRET")}
    secrets.update(AWS_ACCESS_KEY_ID=secrets["CW_KEY_ID"], AWS_SECRET_ACCESS_KEY=secrets["CW_KEY_SECRET"])
    command = (
        'cd "$IRIS_WORKDIR"; exec "$IRIS_PYTHON" -u -m '
        "experiments.post_training.identity_fullscreen_owner.math_only "
        + shlex.join(["--candidate", path, "--candidate-sha256", digest])
    )
    return dict(
        name=name,
        entrypoint=Entrypoint(command=["bash", "-c", command], workdir_files={ARCHIVE_PATH: archive}),
        resources=ResourceSpec(cpu=8, memory="32g", disk="50g"),
        constraints=convert_constraints(
            ResourceConfig.with_cpu(cpu=8, ram="32g", disk="50g", target_cluster="cw-rno2a")
        ),
        replicas=1,
        environment=EnvironmentSpec(
            extras=("cpu",),
            secrets=secrets,
            secret_behavior=SecretBehavior.ISOLATE,
            env_vars={"JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        ),
        timeout=Duration.from_seconds(OWNER_SECONDS),
        max_retries_failure=0,
        max_retries_preemption=0,
        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    candidate = load_candidate(args.candidate, args.candidate_sha256)
    if args.inspect_only:
        print(candidate["label"])
        return
    check_model(candidate)
    run(candidate, ARCHIVE_PATH, ARCHIVE_SHA256)


if __name__ == "__main__":
    main()
