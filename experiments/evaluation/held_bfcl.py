# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Held BFCL launch policy: native evaluation, twelve-hour jobs and zero retries."""

import argparse
import contextlib
import json
import time
from dataclasses import replace
from pathlib import Path

from fray.current_client import current_client
from fray.types import Entrypoint as FrayEntrypoint
from fray.types import JobRequest, JobStatus
from iris.cli.connect import open_iris_client
from iris.client.client import iris_ctx
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2
from marin.evaluation import runner
from marin.evaluation.eval_env import EVAL_ENV_KEYS, EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.evaluation.harbor.runner import canonical_served_name
from marin.evaluation.hardware import Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference import iris as inference
from marin.inference.backend import OPENAI_API_SUFFIX
from marin.inference.config import EffectiveServing
from marin.inference.types import OpenAIEndpoint, RunningModel
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging
from rigging.secrets import resolve_secret_spec
from rigging.timing import Deadline, Duration

from experiments.evaluation.evals import HarborDefinition
from experiments.evaluation.launch import LaunchSpec, prepare_evaluation_batch

TIMEOUT_HOURS = 12
READINESS_SECONDS = 8 * 3600


def wait_for_endpoint(job, endpoint_name):
    """Allow the queued serve a generous readiness window without capacity probing."""
    deadline = Deadline.from_seconds(READINESS_SECONDS)
    while not deadline.expired():
        endpoints = iris_ctx().client.list_endpoint_instances(endpoint_name)
        if endpoints:
            return endpoints[0].address, dict(endpoints[0].metadata)
        if JobStatus.finished(job.status()):
            raise RuntimeError("Inference job finished before endpoint registration")
        time.sleep(2)
    raise TimeoutError("Inference endpoint did not register within eight hours")


@contextlib.contextmanager
def direct_inference(config):
    """Use the native direct inference path with explicit job lifetime policy."""
    if config.instances != 1 or config.broker is not None:
        raise ValueError("This launch binds one direct BFCL serve")
    model = config.model
    iris = replace(
        config.iris, max_retries_failure=0, max_retries_preemption=0, endpoint_ready_timeout_seconds=READINESS_SECONDS
    )
    run_id, endpoint_name = inference._new_endpoint_identity()
    service = inference.IrisServiceConfig(
        model=model,
        engine=config.engine,
        iris=iris,
        endpoint_name=endpoint_name,
        timeout_hours=TIMEOUT_HOURS,
        port_name=None,
    )
    job = current_client().submit(
        JobRequest(
            name=f"inference-{run_id}",
            entrypoint=FrayEntrypoint.from_callable(inference.run_iris_service, args=(service,)),
            resources=iris.worker_resources,
            environment=iris.worker_environment,
            max_retries_failure=0,
            max_retries_preemption=0,
            max_task_failures=0,
            priority=iris.priority,
            timeout=Duration.from_hours(TIMEOUT_HOURS),
        )
    )
    try:
        try:
            address, metadata = wait_for_endpoint(job, endpoint_name)
            tensor_parallel_size = int(metadata[inference._METADATA_TENSOR_PARALLEL_SIZE])
            backend_name = metadata[inference._METADATA_BACKEND]
            effective_serving = EffectiveServing(
                tensor_parallel_size=tensor_parallel_size,
                data_parallel_size=int(metadata[inference._METADATA_DATA_PARALLEL_SIZE]),
                pipeline_parallel_size=int(metadata[inference._METADATA_PIPELINE_PARALLEL_SIZE]),
                task_count=int(metadata[inference._METADATA_TASK_COUNT]),
                max_model_len=(
                    int(metadata[inference._METADATA_MAX_MODEL_LEN])
                    if metadata[inference._METADATA_MAX_MODEL_LEN]
                    else None
                ),
            )
        except Exception as error:
            raise inference.RemoteInferenceStartupError(
                f"Inference job {job.job_id} failed to register a usable endpoint: {error}",
                jobs=(job,),
            ) from error
        running_model = RunningModel(
            endpoint=OpenAIEndpoint(base_url=f"{address.rstrip('/')}{OPENAI_API_SUFFIX}", model=model.model_id),
            tokenizer=model.tokenizer,
        )
        exposed_model = (
            inference._capability_model(running_model, endpoint_name, config.capability_origin)
            if config.capability_origin is not None
            else running_model
        )
        yield inference.RemoteInferenceSession(
            model=exposed_model,
            jobs=(job,),
            endpoint_name=endpoint_name,
            endpoint_health_timeout_seconds=iris.endpoint_health_timeout_seconds,
            streaming=True,
            tensor_parallel_size=tensor_parallel_size,
            backend_name=backend_name,
            effective_serving=effective_serving,
            metrics_url=f"{inference._server_root(exposed_model)}/metrics" if backend_name == "vllm" else None,
        )
    finally:
        inference._terminate_job(job)


def run_batch(batch):
    """Run the unchanged native batch evaluator with the bounded serve lifecycle."""
    configure_coreweave_s3()
    if not batch.evaluations or batch.judge is not None:
        raise ValueError("This entrypoint binds BFCL without a hosted judge")
    orchestrator_job_id = str(iris_ctx().job_id)
    runtime_env = env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS)
    evaluation_env = {**runtime_env, **env_vars_from_keys(tuple(batch.secret_env))}
    config = inference_config_for_model(
        batch.model,
        batch.accelerator,
        env_vars=runtime_env,
        capability_origin=batch.capability_origin,
        api_model=batch.api_model,
        priority=batch.priority_band,
    )
    try:
        runner.precache_inference_model(config)
    except Exception as error:
        runner._record_unstarted(batch, batch.evaluations, error, {"orchestrator": orchestrator_job_id}, {}, [])
        raise
    try:
        with direct_inference(config) as session:
            return runner._evaluate_with_hosted_judge(
                batch,
                session,
                orchestrator_job_id,
                runtime_env,
                evaluation_env,
            )
    except inference.RemoteInferenceStartupError as error:
        runner._record_startup_failure(batch, orchestrator_job_id, error, "inference")
        raise RuntimeError(f"evaluation batch inference failed: {error}") from error


def submit_batch(batch, client):
    """Keep native coordinator resources and environment, binding timeout and retries."""
    constraints = [Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value="cw-rno2a")]
    launch_env = env_vars_from_keys(EVAL_ENV_KEYS)
    for name, spec in sorted(batch.secret_env.items()):
        launch_env[name] = resolve_secret_spec(spec).value
    job = client.submit(
        entrypoint=Entrypoint.from_callable(run_batch, batch),
        name=f"eval-{batch.group_id}",
        resources=ResourceSpec(
            cpu=runner._ORCHESTRATOR_CPU, memory=runner._ORCHESTRATOR_MEMORY, disk=runner._ORCHESTRATOR_DISK
        ),
        environment=EnvironmentSpec(env_vars=launch_env),
        constraints=constraints,
        max_retries_failure=0,
        max_retries_preemption=0,
        timeout=Duration.from_hours(TIMEOUT_HOURS),
        priority_band=batch.priority_band,
    )
    return runner.SubmittedEvaluationBatch(
        group_id=batch.group_id,
        job=job,
        records_prefix=batch.records_prefix,
        model_name=batch.model.name,
        evaluations=tuple(
            runner.SubmittedEvaluation(run_id=evaluation.identity.run_id, eval_name=evaluation.identity.eval_ref.name)
            for evaluation in batch.evaluations
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--harbor-config", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--records-prefix", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_logging()
    spec = LaunchSpec(
        model=load_model_config(args.model_config),
        evals=(),
        evalchemy_definitions=(),
        limit=None,
        harbor_definitions=(
            HarborDefinition(name=canonical_served_name(args.harbor_config.stem), config_path=args.harbor_config),
        ),
        platform=Platform.GPU,
        accelerator="H100x8",
        records_prefix=args.records_prefix,
        submission_cluster="marin",
        federated_cluster="cw-rno2a",
        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
        version=args.version,
        description=args.description,
    )
    batch = prepare_evaluation_batch(spec)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "group_id": batch.group_id,
                    "model": batch.model.name,
                    "model_location": batch.model.location,
                    "evals": [evaluation.identity.eval_ref.name for evaluation in batch.evaluations],
                    "timeout_hours": TIMEOUT_HOURS,
                    "readiness_seconds": READINESS_SECONDS,
                    "failure_retries": 0,
                    "preemption_retries": 0,
                    "priority": "interactive",
                    "target_cluster": "cw-rno2a",
                },
                indent=2,
            )
        )
        return
    with open_iris_client(cluster_name="marin", workspace=Path.cwd()) as client:
        result = submit_batch(batch, client)
        print(f"submitted group {result.group_id}; coordinator {result.job}")
        for evaluation in result.evaluations:
            print(f"{evaluation.run_id} ({evaluation.eval_name})")


if __name__ == "__main__":
    main()
