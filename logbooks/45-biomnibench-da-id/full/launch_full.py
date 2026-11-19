# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public-50 Qwen pass with one shared judge and an eight-hour parent cap."""

import json
import os
import sys
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path

from iris.cli.connect import open_iris_client
from iris.client.client import iris_ctx
from iris.rpc import job_pb2
from marin.evaluation.hardware import Platform
from marin.evaluation.model_config import load_model_config
from rigging.secrets import require_secret
from rigging.timing import Duration

from experiments.evaluation.evals import HarborDefinition
from experiments.evaluation.launch import LaunchSpec, launch_group, prepare_evaluation_batch

root = Path(__file__).resolve().parent
os.environ["DAYTONA_TARGET"] = "us"
assert len(list((root / "runtime-tasks").glob("*/task.toml"))) == 50
assert (root / "runtime-tasks/da-4-6").exists()
gpt_oss = "--gpt-oss" in sys.argv
candidate = "gpt-oss-20b" if gpt_oss else "qwen"
model_path = (
    root.parents[2] / "experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/model-configs/openai-gpt-oss-20b.yaml"
    if gpt_oss else root.parent / "pilot/qwen.yaml"
)
policy_path = root / ("gpt-oss-20b.yaml" if gpt_oss else "full.yaml")
spec = LaunchSpec(
    model=load_model_config(model_path),
    evals=(HarborDefinition(name=f"biomnibench-public50-{candidate}", config_path=policy_path),),
    platform=Platform.GPU,
    accelerator="H100x1" if gpt_oss else "H100x4",
    limit=None,
    records_prefix=None,
    submission_cluster="marin",
    federated_cluster="cw-us-east-02a",
    priority_band=job_pb2.PRIORITY_BAND_INHERIT,
    judge_model=load_model_config(root.parent / "pilot/judge.yaml"),
    judge_accelerator="H100x8",
    version="biomnibench-da-public50-signed-v2",
    description=(
        f"One fresh public-50 {candidate} pass; five disclosed input variants; Pi 0.87.0; "
        "3600s agent; 900s verifier; 16 concurrent; zero solver retries; "
        "at most one malformed-judge retry preserving both responses; "
        "one shared MiniMax judge; eight-hour cap; proposal evidence, not adopted ID policy"
    ),
)
batch = prepare_evaluation_batch(spec)
remote = "--submit-in-cluster" in sys.argv
if "--submit" not in sys.argv and not remote:
    print(
        json.dumps(
            {
                "status": "preflight passed; no submission",
                "tasks": sorted(p.name for p in (root / "runtime-tasks").iterdir()),
                "group_id": batch.group_id,
            },
            indent=2,
        )
    )
    raise SystemExit(0)
assert not (root / "full-evaluation-submission.json").exists(), "Submission already recorded; inspect before any retry"
if remote:
    os.environ["DAYTONA_API_KEY"] = require_secret("DAYTONA_API_KEY")
    # The outer CPU job already federated; children stay on its destination.
    batch = replace(batch, submission_cluster=batch.accelerator.target_cluster)
connection = (
    nullcontext(iris_ctx().client) if remote else open_iris_client(cluster_name="marin", workspace=root.parents[2])
)
with connection as client:
    # Bound this standard launch_group submission without changing evaluator behavior.
    client.submit = partial(
        client.submit,
        timeout=Duration.from_seconds(28800),
        scheduling_timeout=Duration.from_seconds(1800),
        max_retries_preemption=0,
    )
    group = launch_group(batch, client)
    result = {
        "job_id": str(group.job.job_id),
        "group_id": group.group_id,
        "run_ids": [x.run_id for x in group.evaluations],
        "records_prefix": group.records_prefix,
        "parent_timeout_seconds": 28800,
        "scheduling_timeout_seconds": 1800,
        "task_count": 50,
    }
    (root / "full-evaluation-submission.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if remote:
        (Path(os.environ["IRIS_OUTPUT_DIR"]) / "full-evaluation-submission.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        sys.stdout.flush()
        group.job.wait(timeout=28800, poll_interval=60)
