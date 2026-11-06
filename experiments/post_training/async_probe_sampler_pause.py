# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One fixed-weight control of the qualified eight-H100 sampler's busy pause."""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from google.protobuf.duration_pb2 import Duration
from iris.cli.connect import open_iris_client
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, gpu_device
from iris.rpc import job_pb2

CONFIG_URI = (
    "s3://marin-us-east-08a/marin/users/romain/checkpoints/async-probe-cost/"
    "grug-pause-diagnostic-i10-seed1701/2026.11.05.2/exports/probe-cost/config.json"
)
CONFIG_SHA256 = "95c2f9ff20d4575e97c04d5194b330b9c42b0a4d76f5df14045664401b1efa01"
OUTPUT_URI = "s3://marin-us-east-08a/marin/users/romain/research/async-probe-cost-2026.11.05/sampler-pause-control-v1"
JOB_NAME = "async-probe-sampler-pause-control-01a1aa1b-v1"
DATA_URI = "s3://marin-rno2a/marin/users/romain/documents/sc-default-study/2026.10.28.1"
DATA_SHA256 = "54e40005a259f34d5234cc864bc3284bbb8a9db5ff0154b430a6d26553bb40ce"
TOKENIZER_ID = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.10.05"
TOKENIZER_REVISION = "8c76b22afc38f5e95360c83ca88008f00b5ae1e0"


def submit():

    sys.path.insert(0, str(Path("vendor/skyrl").resolve()))
    from cloud.iris.runtime_environment import (  # noqa: PLC0415 - use the vendored source path above
        IRIS_TASK_IMAGE,
        MARINSKYRL_ACTIVATION_FILE,
        RuntimeProfile,
        task_setup_script,
    )

    resources = ResourceSpec(cpu=64, memory="1800GB", disk="2TB")
    resources.device = gpu_device("H100", 8)
    command = (
        f"source {shlex.quote(MARINSKYRL_ACTIVATION_FILE)}\n"
        "cd /app\n"
        "exec python -m experiments.post_training.async_probe_sampler_pause_measure"
    )
    environment = EnvironmentSpec(
        extras=["gpu"],
        setup_scripts=[task_setup_script(RuntimeProfile.MEGATRON)],
        env_vars={
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "ASYNC_PROBE_PAUSE_DIAGNOSTIC": "1",
            "TOGETHER_API_KEY": os.environ["TOGETHER_API_KEY"],
        },
    )
    with open_iris_client(config_file=Path("lib/iris/config/marin.yaml"), workspace=Path.cwd()) as client:
        job = client.submit(
            entrypoint=Entrypoint.from_command("bash", "-lc", command),
            name=JOB_NAME,
            resources=resources,
            environment=environment,
            task_image=IRIS_TASK_IMAGE,
            constraints=[Constraint.create(CLUSTER_CONSTRAINT_KEY, ConstraintOp.EQ, value="cw-rno2a")],
            priority_band=job_pb2.PRIORITY_BAND_BATCH,
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
            replicas=1,
            max_retries_failure=0,
            max_retries_preemption=0,
            max_task_failures=0,
            timeout=Duration(seconds=1800),
        )
        print(f"Submitted sampler-only control: {job.job_id}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.submit:
        submit()
    else:
        print(
            json.dumps(
                {
                    "job_name": JOB_NAME,
                    "gpus": 8,
                    "nodes": 1,
                    "timeout_seconds": 1800,
                    "failure_retries": 0,
                    "preemption_retries": 0,
                    "config_sha256": CONFIG_SHA256,
                }
            )
        )
