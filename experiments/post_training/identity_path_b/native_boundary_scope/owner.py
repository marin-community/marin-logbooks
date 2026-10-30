# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Own one original-template native-boundary screen without output repair."""

import argparse
import hashlib
import json
import re
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
from experiments.post_training.identity_path_b import trend_serve
from experiments.post_training.identity_path_b.async_launch import (
    CPU_RESOURCES,
    checked_export_event,
    regional_task_context,
)
from experiments.post_training.identity_path_b.checkpoints import TEMPORARY_PREFIX, CheckpointEvent
from experiments.post_training.identity_path_b.identity_scope import run, source_baseline
from experiments.post_training.identity_path_b.native_boundary_scope.run import NativeSession
from experiments.post_training.identity_path_b.trend import wait_for_owned_endpoint
from experiments.post_training.identity_path_b.workflow import ROOT_LIFETIME
from experiments.post_training.identity_t2 import serve as t2

TARGETS_SHA256 = "12c5dabb4ee1cb9b527f65f4a9de5b70eb8646eaca0d02200e5d4916ca484670"


def checked_audit_sha256(value):
    """Bind the actual completed audit, never a future or guessed artifact."""
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("Require actual HF audit SHA256 from the executor readiness receipt")
    return value


def target(label):
    raw = Path(__file__).with_name("targets.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != TARGETS_SHA256:
        raise ValueError("Committed native-boundary targets changed")
    return json.loads(raw)[label]


def checked_identity(label, name, audit_sha256) -> dict:
    item = target(label)
    if label == "v01":
        identity = source_baseline.checked_source(name)
        raw_config = (StoragePath(item["model"]) / "tokenizer_config.json").read_bytes()
        identity["actual_tokenizer_config_sha256"] = hashlib.sha256(raw_config).hexdigest()
        identity["effective_prefix_admission"] = (
            "Every CPU/server/returned prefix must equal the original132 IDs; no template override"
        )
        return identity
    event = CheckpointEvent(**item["event"])
    checked_export_event(event)
    raw_audit = (StoragePath(event.hf_export) / "path-b-storage-audit.json").read_bytes()
    if hashlib.sha256(raw_audit).hexdigest() != checked_audit_sha256(audit_sha256):
        raise ValueError("Audited HF tensor/storage admission changed")
    audit = json.loads(raw_audit)
    if (
        len(audit["tensors"]) != 502
        or audit["frozen_native_tensors"] != 104
        or audit["trained_master_cast_tensors"] != 398
    ):
        raise ValueError("Require exact audited104 frozen/398 cast-master HF")
    expected = {
        "tokenizer.json": "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45",
        "tokenizer_config.json": "d9270d9407a15ec895fa2cdb0b6fbe9cd7dab937b91c2db708a918b35efa18b3",
        "chat_template.jinja": "b3f20bc21498407f2815dd1a6c26150192926ccf04f6324ca40a1209bff7d1de",
    }
    for filename, digest in expected.items():
        if hashlib.sha256((StoragePath(event.hf_export) / filename).read_bytes()).hexdigest() != digest:
            raise ValueError("Native trained-HF tokenizer metadata changed: " + filename)
    return {"event": event.payload(), "actual_tokenizer_metadata": expected}


def service_config(label, name):
    if label == "v01":
        return source_baseline.service_config(name)
    return replace(trend_serve.service_config(CheckpointEvent(**target(label)["event"])), timeout_hours=12)


def endpoint(label, name):
    if label == "v01":
        return source_baseline.source_identity(name)["endpoint_name"]
    return "/serve/" + trend_serve.event_identity(CheckpointEvent(**target(label)["event"]))


def serve_submission(label, name, credentials, audit_sha256, readiness_sha256):
    request = t2.submission(credentials)
    script = (
        'set -euo pipefail\nroot_python="$IRIS_VENV/bin/python"\n'
        "source /app/vendor/skyrl/.iris-runtime-env\n"
        'exec "$root_python" -u -m experiments.post_training.identity_path_b.native_boundary_scope.owner '
        + shlex.join(
            [
                "--target",
                label,
                "--name",
                name,
                "--audit-sha256",
                checked_audit_sha256(audit_sha256),
                "--readiness-sha256",
                checked_audit_sha256(readiness_sha256),
                "--serve",
            ]
        )
    )
    request.update(
        name=name + "-serve",
        entrypoint=Entrypoint(command=["bash", "-c", script]),
        timeout=Duration.from_hours(12),
        constraints=None,
    )
    return request


def screen_worker(label, name, output, audit_sha256, readiness_sha256):
    item = target(label)
    if not name.startswith(f"pathb-native132-{label.lower()}-r1-") or not output.startswith(TEMPORARY_PREFIX):
        raise ValueError("Require fresh authorized native-boundary target namespace")
    destination = StoragePath(output)
    if destination.exists():
        raise FileExistsError("Native-boundary output exists; no automatic retry")
    identity = checked_identity(label, name, audit_sha256)
    identity["bound_actual_storage_audit_sha256"] = checked_audit_sha256(audit_sha256)
    identity["actual_executor_readiness_receipt_sha256"] = checked_audit_sha256(readiness_sha256)
    (destination / "input-identity.json").write_bytes((json.dumps(identity, indent=2) + "\n").encode())
    context = regional_task_context()
    assert context.client is not None
    credentials = {key: require_secret(key) for key in ("CW_KEY_ID", "CW_KEY_SECRET")}
    server = context.client.submit(**serve_submission(label, name, credentials, audit_sha256, readiness_sha256))
    status = {"target": label, "serve_job": str(server.job_id), "status": "pending"}
    try:
        base_url = wait_for_owned_endpoint(server, endpoint(label, name))
        with run.payload() as (root, historical):

            def persist(index, record):
                (destination / f"raw/{index:03d}.json").write_bytes(
                    (json.dumps(record, ensure_ascii=False) + "\n").encode()
                )

            records = run.generate(base_url, item["model"], root, historical, persist, session_factory=NativeSession)
        (destination / "generations.jsonl").write_bytes(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode()
        )
        status.update(
            status="complete",
            records=len(records),
            responses=sum(row["coverage"] == "response_received" for row in records),
            errors=sum(row["coverage"] != "response_received" for row in records),
            semantic_status="UNREAD_NATIVE_DIAGNOSTIC",
        )
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        try:
            (destination / "status.json").write_bytes((json.dumps(status, indent=2) + "\n").encode())
        finally:
            server.cancel()


def submission(*, label, name, output, credentials, audit_sha256, readiness_sha256):
    target(label)
    audit_sha256 = checked_audit_sha256(audit_sha256)
    readiness_sha256 = checked_audit_sha256(readiness_sha256)
    if not name.startswith(f"pathb-native132-{label.lower()}-r1-") or not output.startswith(TEMPORARY_PREFIX):
        raise ValueError("Require authorized fresh native-boundary namespace")
    secrets = {key: credentials[key] for key in ("CW_KEY_ID", "CW_KEY_SECRET")}
    secrets.update(AWS_ACCESS_KEY_ID=secrets["CW_KEY_ID"], AWS_SECRET_ACCESS_KEY=secrets["CW_KEY_SECRET"])
    return dict(
        name=name,
        entrypoint=Entrypoint.from_command(
            "bash",
            "-c",
            'exec "$IRIS_PYTHON" -u -m experiments.post_training.identity_path_b.native_boundary_scope.owner '
            + shlex.join(
                [
                    "--target",
                    label,
                    "--name",
                    name,
                    "--output",
                    output,
                    "--audit-sha256",
                    audit_sha256,
                    "--readiness-sha256",
                    readiness_sha256,
                ]
            ),
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
    parser.add_argument("--target", choices=("A9", "A10"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output")
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--readiness-sha256", required=True)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    checked_audit_sha256(args.audit_sha256)
    checked_audit_sha256(args.readiness_sha256)
    if args.inspect_only:
        with run.payload() as (root, historical):
            print(
                json.dumps(
                    dict(
                        target=args.target,
                        model=target(args.target)["model"],
                        records=len(run.jobs(root, historical)),
                        calls=0,
                    )
                )
            )
        return
    if args.serve:
        checked_identity(args.target, args.name, args.audit_sha256)
        t2.validate_runtime()
        t2.configure_worker_s3()
        run_iris_service(service_config(args.target, args.name))
        return
    if args.output is None:
        raise ValueError("CPU owner requires output")
    screen_worker(args.target, args.name, args.output, args.audit_sha256, args.readiness_sha256)


if __name__ == "__main__":
    main()
