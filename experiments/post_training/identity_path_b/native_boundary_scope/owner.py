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
from experiments.post_training.identity_path_b.async_launch import (
    CPU_RESOURCES,
    regional_task_context,
)
from experiments.post_training.identity_path_b.checkpoints import TEMPORARY_PREFIX
from experiments.post_training.identity_path_b.identity_scope import run
from experiments.post_training.identity_path_b.native_boundary_scope.run import NativeSession
from experiments.post_training.identity_path_b.replay_serve import T2_SERVE_SHA256
from experiments.post_training.identity_path_b.trend import wait_for_owned_endpoint
from experiments.post_training.identity_path_b.workflow import ROOT_LIFETIME
from experiments.post_training.identity_t2 import serve as t2

TARGETS_SHA256 = "3f5a0d785518e9ec73b8776efc6d282a4420589148a3dcc51db8c8b54d86d4b9"


def checked_audit_sha256(value):
    """Bind the actual completed audit, never a future or guessed artifact."""
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("Require actual merge admission SHA256 from the executor readiness receipt")
    return value


def target(label):
    raw = Path(__file__).with_name("targets.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != TARGETS_SHA256:
        raise ValueError("Committed native-boundary targets changed")
    return json.loads(raw)[label]


def checked_identity(label, name, audit_sha256) -> dict:
    """Admit an actual merged HF without inventing a training/export event."""
    item = target(label)
    if checked_audit_sha256(audit_sha256) != item["merge_admission_sha256"]:
        raise ValueError("Require this exact completed merge admission")
    raw = StoragePath(item["merge_admission_uri"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != audit_sha256:
        raise ValueError("Actual merge admission changed")
    report = json.loads(raw)
    if (
        report["admitted"] is not True
        or report["provenance_kind"] != "linear-cpu-merge"
        or report["label"] != label
        or report["output"] != item["model"]
        or report["code_revision"] != item["merge_code_revision"]
        or report["training_event"] is not None
        or report["master_cast_audit"] is not None
        or report["training_checkpoint_provenance"] is not False
        or len(report["tensors"]) != 502
        or report["source_frozen_fields"] != 104
        or len(report["router_biases_fp32"]) != 26
        or report["protected_rows_per_tensor"] != 256
        or report["recipe"]["parameters"]
        != {"coefficients": [0.51], "density": 1, "method": "average", "scale": 1, "seed": 0}
        or report["pre_arithmetic_parent_protection"]["full_fields"] != 104
        or report["pre_arithmetic_parent_protection"]["router_biases_fp32"] != 26
        or len(report["pre_arithmetic_parent_protection"]["parent_bits_admitted_before_restoration"]) != 106
    ):
        raise ValueError("Require the exact finite/restored502 M51 merge report")
    for name in report["router_biases_fp32"]:
        if report["tensors"][name]["dtype"] != "torch.float32":
            raise ValueError("Merge router bias must remain FP32")
    for parent in ("A3", "A8a"):
        if len(report["parents_full_hash_readback"][parent]) != 56:
            raise ValueError("Require full parent object hashes")
    for parent in ("A3", "A8a"):
        if report["subsets"]["frozen104"]["delta_l2_vs_" + parent] != 0:
            raise ValueError("Protected frozen fields changed")
    manifest_raw = (StoragePath(item["model"]) / "merge-manifest.json").read_bytes()
    if (
        hashlib.sha256(manifest_raw).hexdigest() != item["merge_manifest_sha256"]
        or report["merge_manifest_sha256"] != item["merge_manifest_sha256"]
    ):
        raise ValueError("Actual merge manifest changed")
    for filename, digest in item["metadata_sha256"].items():
        if hashlib.sha256((StoragePath(item["model"]) / filename).read_bytes()).hexdigest() != digest:
            raise ValueError("Actual merge HF metadata changed: " + filename)
    return {
        "kind": "linear-cpu-merge",
        "model": item["model"],
        "actual_merge_admission_sha256": audit_sha256,
        "actual_merge_manifest_sha256": item["merge_manifest_sha256"],
        "training_event": None,
        "master_cast_audit": None,
        "actual_HF_metadata_sha256": item["metadata_sha256"],
    }


def service_config(label, name):
    if hashlib.sha256(Path(t2.__file__).read_bytes()).hexdigest() != T2_SERVE_SHA256:
        raise ValueError("The frozen native T2 serve source changed")
    original = t2.service_config()
    return replace(
        original,
        model=replace(original.model, weights=target(label)["model"]),
        iris=replace(original.iris, priority=job_pb2.PRIORITY_BAND_PRODUCTION),
        endpoint_name=endpoint(label, name),
        timeout_hours=12,
        controller_proxy_timeout_seconds=1800,
    )


def endpoint(label, name):
    target(label)
    return "/serve/" + name + "-serve"


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
                "--merge-admission-sha256",
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
        priority_band=job_pb2.PRIORITY_BAND_PRODUCTION,
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
    identity["bound_actual_merge_admission_sha256"] = checked_audit_sha256(audit_sha256)
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
                    "--merge-admission-sha256",
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
        priority_band=job_pb2.PRIORITY_BAND_PRODUCTION,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("M51",), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output")
    parser.add_argument("--merge-admission-sha256", required=True)
    parser.add_argument("--readiness-sha256", required=True)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    checked_audit_sha256(args.merge_admission_sha256)
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
        checked_identity(args.target, args.name, args.merge_admission_sha256)
        t2.validate_runtime()
        t2.configure_worker_s3()
        run_iris_service(service_config(args.target, args.name))
        return
    if args.output is None:
        raise ValueError("CPU owner requires output")
    screen_worker(args.target, args.name, args.output, args.merge_admission_sha256, args.readiness_sha256)


if __name__ == "__main__":
    main()
