# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Three-task diagnostic through the standard launcher, with a four-hour parent cap."""

import json
import os
import sys
from functools import partial
from pathlib import Path

from iris.cli.connect import open_iris_client
from iris.rpc import job_pb2
from marin.evaluation.hardware import Platform
from marin.evaluation.model_config import load_model_config
from rigging.timing import Duration

from experiments.evaluation.evals import HarborDefinition
from experiments.evaluation.launch import LaunchSpec, launch_group, prepare_evaluation_batch

root = Path(__file__).resolve().parent
os.environ["DAYTONA_TARGET"] = "us"
assert len(list((root / "remaining-tasks").glob("*/task.toml"))) == 3
assert not (root / "remaining-tasks/da-4-6").exists()
spec = LaunchSpec(
    model=load_model_config(root / "qwen.yaml"),
    evals=(HarborDefinition(name="biomnibench-remaining-pilot", config_path=root / "remaining.yaml"),),
    platform=Platform.GPU,
    accelerator="H100x4",
    limit=None,
    records_prefix=None,
    submission_cluster="marin",
    federated_cluster="cw-us-east-02a",
    priority_band=job_pb2.PRIORITY_BAND_INHERIT,
    judge_model=load_model_config(root / "judge.yaml"),
    judge_accelerator="H100x8",
    version="biomnibench-da-id-proposal-pilot",
    description=(
        "Three remaining diagnostic tasks: da-1-3, da-17-3, da-20-3; Pi 0.87.0;"
        " 3600s agent; 900s verifier; two concurrent; zero retries; four-hour o"
        "rchestrator cap; separate from completed control; not official ID or p"
        "aper parity"
    ),
)
batch = prepare_evaluation_batch(spec)
if "--submit" not in sys.argv:
    print(
        json.dumps(
            {
                "status": "preflight passed; no submission",
                "tasks": ["da-1-3", "da-17-3", "da-20-3"],
                "group_id": batch.group_id,
            },
            indent=2,
        )
    )
    raise SystemExit(0)
assert not (
    root / "remaining-evaluation-submission.json"
).exists(), "Submission already recorded; inspect before any retry"
with open_iris_client(cluster_name="marin", workspace=root.parents[2]) as client:
    # Bound this standard launch_group submission without changing evaluator behavior.
    client.submit = partial(
        client.submit,
        timeout=Duration.from_seconds(14400),
        scheduling_timeout=Duration.from_seconds(1800),
        max_retries_preemption=0,
    )
    group = launch_group(batch, client)
    result = {
        "job_id": str(group.job.job_id),
        "group_id": group.group_id,
        "run_ids": [x.run_id for x in group.evaluations],
        "records_prefix": group.records_prefix,
        "parent_timeout_seconds": 14400,
        "scheduling_timeout_seconds": 1800,
        "task_count": 3,
    }
    (root / "remaining-evaluation-submission.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
