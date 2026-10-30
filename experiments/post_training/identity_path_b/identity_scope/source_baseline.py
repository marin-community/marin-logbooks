# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Original identity132 on published v0.1, with honest source-only provenance."""

import argparse
import hashlib
import json
import shlex
from dataclasses import replace
from pathlib import Path

from fray.iris_backend import convert_constraints, convert_resources
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from marin.inference.iris import run_iris_service
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import SecretBehavior, require_secret
from rigging.timing import Duration

from experiments.post_training.identity_context_distill.runtime_recipe import xla_precision_env
from experiments.post_training.identity_path_b.async_launch import CPU_RESOURCES, regional_task_context
from experiments.post_training.identity_path_b.checkpoints import TEMPORARY_PREFIX
from experiments.post_training.identity_path_b.identity_scope.run import generate, jobs, payload
from experiments.post_training.identity_path_b.launch import SOURCE_OBJECTS, SOURCE_RECEIPT
from experiments.post_training.identity_path_b.replay_serve import T2_SERVE_SHA256
from experiments.post_training.identity_path_b.trend import wait_for_owned_endpoint
from experiments.post_training.identity_path_b.workflow import ROOT_LIFETIME
from experiments.post_training.identity_t2 import serve as t2
from experiments.post_training.identity_t2.capture import TEMPLATE_SHA256, TOKENIZER_SHA256

CHILD_LIFETIME = 12 * 3600


def source_identity(name: str) -> dict:
    return {
        "kind": "PUBLISHED_V01_SOURCE_NOT_A_TRAINED_PATHB_CHECKPOINT",
        "model": t2.MODEL,
        "source_revision": "adc0078a19a4383bc998fcdb43867b3360baf5f3",
        "source_receipt": SOURCE_RECEIPT,
        "source_objects": SOURCE_OBJECTS,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "template_sha256": TEMPLATE_SHA256,
        "endpoint_name": "/serve/" + name + "-serve",
        "no_checkpoint_event_or_retirement_or_export_audit": True,
    }


def checked_source(name: str) -> dict:
    identity = source_identity(name)
    metadata = {}
    for label, item in (("source_receipt", SOURCE_RECEIPT), ("source_objects", SOURCE_OBJECTS)):
        raw = StoragePath(item["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("Published source provenance changed")
        metadata[label] = {"sha256": item["sha256"], "bytes": len(raw)}
    for filename, digest in (("tokenizer.json", TOKENIZER_SHA256), ("chat_template.jinja", TEMPLATE_SHA256)):
        raw = (StoragePath(t2.MODEL) / filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Published source tokenizer/template changed")
        metadata[filename] = {"sha256": digest, "bytes": len(raw)}
    identity["actual_metadata_full_get"] = metadata
    return identity


def service_config(name: str):
    if hashlib.sha256(Path(t2.__file__).read_bytes()).hexdigest() != T2_SERVE_SHA256:
        raise ValueError("Frozen native T2 service changed")
    original = t2.service_config()
    return replace(
        original,
        endpoint_name=source_identity(name)["endpoint_name"],
        timeout_hours=12,
        controller_proxy_timeout_seconds=1800,
    )


def serve_submission(name: str, credentials: dict[str, str]) -> dict:
    request = t2.submission(credentials)
    script = (
        'set -euo pipefail\nroot_python="$IRIS_VENV/bin/python"\n'
        "source /app/vendor/skyrl/.iris-runtime-env\n"
        'exec "$root_python" -u -m experiments.post_training.identity_path_b.identity_scope.source_baseline '
        + shlex.join(["--name", name, "--serve"])
    )
    request.update(
        name=name + "-serve",
        entrypoint=Entrypoint(command=["bash", "-c", script]),
        timeout=Duration.from_seconds(CHILD_LIFETIME),
        constraints=None,
    )
    return request


def screen_worker(name: str, output: str) -> None:
    if not name.startswith("pathb-identity-v01-r1-") or not output.startswith(TEMPORARY_PREFIX):
        raise ValueError("Require the fresh source-baseline namespace")
    destination = StoragePath(output)
    if destination.exists():
        raise FileExistsError("Baseline output already exists; no automatic retry")
    identity = checked_source(name)
    (destination / "input-source.json").write_bytes((json.dumps(identity, indent=2) + "\n").encode())
    context = regional_task_context()
    assert context.client is not None
    credentials = {key: require_secret(key) for key in ("CW_KEY_ID", "CW_KEY_SECRET")}
    server = context.client.submit(**serve_submission(name, credentials))
    status = {"source": identity, "serve_job": str(server.job_id), "status": "pending"}
    try:
        base_url = wait_for_owned_endpoint(server, identity["endpoint_name"])
        with payload() as (root, historical):

            def persist(index: int, record: dict):
                (destination / f"raw/{index:03d}.json").write_bytes(
                    (json.dumps(record, ensure_ascii=False) + "\n").encode()
                )

            records = generate(base_url, t2.MODEL, root, historical, persist)
        (destination / "generations.jsonl").write_bytes(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode()
        )
        status.update(
            status="complete",
            records=len(records),
            responses=sum(row["coverage"] == "response_received" for row in records),
            errors=sum(row["coverage"] != "response_received" for row in records),
            semantic_status="UNGRADED_PENDING_ROLE_AWARE_READ",
        )
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        # The original generate() drains every issued future before returning.
        try:
            (destination / "status.json").write_bytes((json.dumps(status, indent=2) + "\n").encode())
        finally:
            server.cancel()


def screen_submission(*, name: str, output: str, credentials: dict[str, str]) -> dict:
    if not name.startswith("pathb-identity-v01-r1-") or not output.startswith(TEMPORARY_PREFIX):
        raise ValueError("Require the authorized fresh v0.1 baseline")
    secrets = {key: credentials[key] for key in ("CW_KEY_ID", "CW_KEY_SECRET")}
    secrets.update(AWS_ACCESS_KEY_ID=secrets["CW_KEY_ID"], AWS_SECRET_ACCESS_KEY=secrets["CW_KEY_SECRET"])
    return dict(
        name=name,
        entrypoint=Entrypoint.from_command(
            "bash",
            "-c",
            'exec "$IRIS_PYTHON" -u -m experiments.post_training.identity_path_b.identity_scope.source_baseline '
            + shlex.join(["--name", name, "--output", output]),
        ),
        resources=convert_resources(CPU_RESOURCES),
        constraints=convert_constraints(CPU_RESOURCES) or None,
        replicas=1,
        environment=EnvironmentSpec(
            extras=["gpu"],
            secrets=secrets,
            secret_behavior=SecretBehavior.ISOLATE,
            env_vars={**xla_precision_env("default"), "JAX_PLATFORMS": "cpu"},
        ),
        timeout=Duration.from_seconds(ROOT_LIFETIME),
        max_retries_failure=0,
        max_retries_preemption=0,
        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.inspect_only:
        with payload() as (root, historical):
            print(json.dumps({"source": source_identity(args.name), "records": len(jobs(root, historical)), "calls": 0}))
        return
    if args.serve:
        checked_source(args.name)
        t2.validate_runtime()
        t2.configure_worker_s3()
        run_iris_service(service_config(args.name))
        return
    if args.output is None:
        raise ValueError("CPU owner needs an output prefix")
    screen_worker(args.name, args.output)


if __name__ == "__main__":
    main()
