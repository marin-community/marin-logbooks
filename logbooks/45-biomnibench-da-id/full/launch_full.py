# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a public-50 candidate with the shared MiniMax judge."""

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
gemma = "--gemma" in sys.argv
assert not (gpt_oss and gemma)
candidate, accelerator = "qwen", "H100x4"
model_path = root.parent / "pilot/qwen.yaml"
policy_path = root / "full.yaml"
if gpt_oss or gemma:
    candidate, accelerator, filename = (
        ("gemma-4-26b-a4b-it", "H100x2", "google-gemma-4-26B-A4B-it.yaml")
        if gemma
        else ("gpt-oss-20b", "H100x1", "openai-gpt-oss-20b.yaml")
    )
    model_path = (
        root.parents[2] / "experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/model-configs" / filename
    )
    policy_path = root / "gpt-oss-20b.yaml"
final_task = "--final-task" in sys.argv
benchmark = f"biomnibench-public50-{candidate}"
if final_task:
    assert gpt_oss or gemma
    policy_path = root / "gpt-oss-final.yaml"
    benchmark = f"biomnibench-final-da-13-1-{candidate}"
cap = 7200 if final_task else 28800
long_queue = "--two-day-window" in sys.argv
if long_queue:
    assert final_task and gemma
    cap = 48 * 3600
scheduling_cap = cap if long_queue else 1800
spec = LaunchSpec(
    model=load_model_config(model_path),
    evals=(HarborDefinition(name=benchmark, config_path=policy_path),),
    platform=Platform.GPU,
    accelerator=accelerator,
    limit=None,
    records_prefix=None,
    submission_cluster="marin",
    federated_cluster="cw-us-east-02a",
    priority_band=job_pb2.PRIORITY_BAND_INHERIT,
    judge_model=load_model_config(root.parent / "pilot/judge.yaml"),
    judge_accelerator="H100x8",
    version="biomnibench-da-public50-signed-v2",
    description=(
        (
            "Replacement trial for da-13-1; Pi 0.87.0; "
            if final_task
            else f"One fresh public-50 {candidate} pass; five disclosed input variants; Pi 0.87.0; "
        )
        + "3600s agent; 900s verifier; 16 concurrent; zero solver retries; "
        + ("two inline attempts and up to two frozen retries; " if gpt_oss or gemma else "one malformed-judge retry; ")
        + "one shared MiniMax judge; bounded parent; proposal evidence, not adopted ID policy"
    ),
)
batch = prepare_evaluation_batch(spec)
if long_queue:
    batch = replace(batch, placement_timeout_seconds=cap)
remote = "--submit-in-cluster" in sys.argv
if "--submit" not in sys.argv and not remote:
    print(
        json.dumps(
            {
                "status": "preflight passed; no submission",
                "tasks": ["da-13-1"] if final_task else sorted(p.name for p in (root / "runtime-tasks").iterdir()),
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
        timeout=Duration.from_seconds(cap),
        scheduling_timeout=Duration.from_seconds(scheduling_cap),
        max_retries_preemption=0,
    )
    group = launch_group(batch, client)
    result = {
        "job_id": str(group.job.job_id),
        "group_id": group.group_id,
        "run_ids": [x.run_id for x in group.evaluations],
        "records_prefix": group.records_prefix,
        "parent_timeout_seconds": cap,
        "scheduling_timeout_seconds": scheduling_cap,
        "task_count": 1 if final_task else 50,
    }
    (root / "full-evaluation-submission.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if remote:
        (Path(os.environ["IRIS_OUTPUT_DIR"]) / "full-evaluation-submission.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        sys.stdout.flush()
        group.job.wait(timeout=cap, poll_interval=60)
