# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent checkpoint diagnostics: fixed66 OFF plus one GSM8K decode.

This CPU coordinator owns only its exact serve and GSM child. It never controls
training or scans capacity. Optional BFCL is left to the executor; no automatic
BFCL launch or final-panel access is included.
"""

import hashlib
import json
import shlex
import time
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.client.client import iris_ctx
from iris.cluster.types import Entrypoint
from iris.resources.state import is_job_finished
from marin.evaluation.evalchemy.client import CONFIG_ENV_KEY
from marin.evaluation.evalchemy.runner import (
    EvalchemyOutcome,
    EvalchemyRunConfig,
    EvalchemyRuntimeConfig,
    _evalchemy_client_command,
    _run_config_json,
    read_evalchemy_outcome,
)
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.inference.types import OpenAIEndpoint, RunningModel
from pydantic import TypeAdapter
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret
from rigging.timing import Deadline

from experiments.post_training.identity_path_b.async_launch import (
    CHILD_LIFETIME,
    CPU_RESOURCES,
    TrendInputs,
    checked_export_event,
    native_submit,
    write_json,
)
from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent, event_bytes
from experiments.post_training.identity_path_b.trend_identity import run_identity_trend
from experiments.post_training.identity_path_b.trend_serve import event_identity, submission


def gsm_config(event: CheckpointEvent) -> EvalchemyRunConfig:
    return EvalchemyRunConfig(
        name="gsm8k_0shot",
        tasks=(EvalTaskConfig(name="gsm8k", num_fewshot=0, task_alias="gsm8k_0shot", generation=True),),
        apply_chat_template=True,
        max_gen_toks=8192,
        max_length=32768,
        num_concurrent=8,
        seed=0,
        extra_gen_kwargs={"temperature": "0"},
        chat_template_kwargs={"enable_thinking": False},
        # Preserve the historical native HTTP policy, separately from job0/0:
        # timeout1800, transport budget5400, attempt timeout1800. The budget,
        # rather than the nominal max_retries8, governs transient attempts.
        runtime=EvalchemyRuntimeConfig(
            extras=("serve-eval",), dataset_cache=f"{event.eval_scratch}/gsm8k-dataset-cache"
        ),
    )


def wait_for_owned_endpoint(job, endpoint_name: str) -> str:
    """Bounded startup discovery of one owned endpoint, without model requests."""
    deadline = Deadline.from_seconds(CHILD_LIFETIME - 3600)
    context = iris_ctx()
    assert context.client is not None
    while True:
        deadline.raise_if_expired("Checkpoint serve did not register before the fixed startup deadline")
        if is_job_finished(job.status().state):
            raise RuntimeError("Owned checkpoint serving job finished before endpoint registration")
        try:
            return context.client.resolve_endpoint(endpoint_name).rstrip("/")
        except ConnectError as error:
            if error.code != Code.NOT_FOUND:
                raise
        time.sleep(min(2, deadline.remaining_seconds()))


def gsm_submission(model: RunningModel, config: EvalchemyRunConfig, output: str):
    # Reuse native client configuration, runtime command and outcome reader.
    # Native run_evalchemy's submit omits preemption/lifetime/priority settings;
    # only this experiment's scoped submit supplies the owner-required policy.
    command = (
        'cd "$IRIS_WORKDIR"\n'
        'export UV_PROJECT_ENVIRONMENT="$IRIS_WORKDIR/vendor/evalchemy/.venv"\n'
        f"exec {shlex.join(_evalchemy_client_command(config.runtime))} "
        '"$IRIS_WORKDIR/experiments/post_training/identity_path_b/trend_gsm_client.py"'
    )
    return native_submit(
        Entrypoint.from_command("bash", "-c", command),
        name="gsm8k-one-decode",
        resources=CPU_RESOURCES,
        replicas=1,
        env_vars={
            "JAX_PLATFORMS": "cpu",
            "HF_ALLOW_CODE_EVAL": "1",
            "OPENAI_API_KEY": "local-endpoint",
            "TQDM_MININTERVAL": "30",
            CONFIG_ENV_KEY: _run_config_json(model, config, output),
        },
        lifetime=CHILD_LIFETIME,
    )


def trend_worker(event: CheckpointEvent, inputs: TrendInputs) -> None:
    """Run after audited export; always retain status and release exact owned serve."""
    checked_export_event(event)
    context = iris_ctx()
    assert context.client is not None
    raw = event_bytes(event)
    event_uri = f"{event.eval_scratch}/input-event.json"
    StoragePath(event_uri).write_bytes(raw)
    record = {
        "status": "pending",
        "event": event.payload(),
        "event_sha256": hashlib.sha256(raw).hexdigest(),
        "identity_profile_status": "UNGRADED_PENDING_ROLE_AWARE_SEMANTIC_READ",
        "bfcl": "not_requested; optional executor decision without capacity census",
        "job_failure_retries": 0,
        "job_preemption_retries": 0,
    }
    server = None
    gsm_job = None
    try:
        # Iris secrets can be mounted files, not os.environ values. Never emit
        # or serialize this live dictionary in a receipt/Entrypoint argument.
        credentials = {key: require_secret(key) for key in ("CW_KEY_ID", "CW_KEY_SECRET")}
        kwargs = submission(event, event_uri, record["event_sha256"], credentials)
        server = context.client.submit(**kwargs)
        record["serve_job"] = str(server.job_id)
        write_json(f"{event.eval_scratch}/trend.json", record)
        base_url = wait_for_owned_endpoint(server, f"/serve/{event_identity(event)}")
        model = RunningModel(OpenAIEndpoint(f"{base_url}/v1", event.hf_export), tokenizer=event.hf_export)
        config = gsm_config(event)
        gsm_job = gsm_submission(model, config, f"{event.eval_scratch}/gsm8k_0shot")
        record["gsm_job"] = str(gsm_job.job_id)
        record["gsm_config"] = asdict(config)
        write_json(f"{event.eval_scratch}/trend.json", record)
        with TemporaryDirectory(prefix="pathb-trend-") as local:
            prompt_dir = Path(local)
            (prompt_dir / "identity.jsonl").write_bytes(StoragePath(inputs.identity_path).read_bytes())
            (prompt_dir / "controls.jsonl").write_bytes(StoragePath(inputs.controls_path).read_bytes())
            rows = run_identity_trend(
                base_url,
                event.hf_export,
                prompt_dir,
                write_record=lambda row: write_json(f"{event.eval_scratch}/identity-raw/{row['row_id']}.json", row),
            )
        generations = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        StoragePath(f"{event.eval_scratch}/generations-66-off.jsonl").write_bytes(generations)
        record["identity_control_coverage"] = {
            "requested": len(rows),
            "received": sum(r["coverage"] == "response_received" for r in rows),
            "errors": sum(r["coverage"] != "response_received" for r in rows),
            "generations_sha256": hashlib.sha256(generations).hexdigest(),
        }
        # This independent eval coordinator may wait for its own GSM child.
        # Neither training nor its export callback waits here.
        gsm_job.wait(timeout=CHILD_LIFETIME)
        outcome = read_evalchemy_outcome(config, f"{event.eval_scratch}/gsm8k_0shot", str(gsm_job.job_id))
        record["gsm_outcome"] = TypeAdapter(EvalchemyOutcome).dump_python(outcome, mode="json")
        coverages = list(outcome.coverage.values())
        record["gsm_coverage_complete"] = len(coverages) == 1 and coverages[0].n_scored == 1319
        record["status"] = (
            "generated_pending_semantic_read"
            if (record["identity_control_coverage"]["errors"] == 0 and record["gsm_coverage_complete"])
            else "incomplete_coverage"
        )
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, application_resubmits=0)
        raise
    finally:
        try:
            write_json(f"{event.eval_scratch}/trend.json", record)
        finally:
            # Release even if the final receipt write itself fails. Completed
            # identity responses were already persisted independently.
            try:
                if gsm_job is not None and not is_job_finished(gsm_job.status().state):
                    gsm_job.cancel()
            finally:
                if server is not None:
                    server.cancel()
