# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One CPU owner holds the exact native serve until its132 response records persist."""

import argparse
import hashlib
import json
import shlex

from fray.iris_backend import convert_constraints, convert_resources
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import SecretBehavior, require_secret
from rigging.timing import Duration

from experiments.post_training.identity_context_distill.runtime_recipe import xla_precision_env
from experiments.post_training.identity_path_b.async_launch import (
    CPU_RESOURCES,
    checked_export_event,
    regional_task_context,
)
from experiments.post_training.identity_path_b.checkpoints import TEMPORARY_PREFIX, CheckpointEvent, event_bytes
from experiments.post_training.identity_path_b.identity_scope.run import generate, payload
from experiments.post_training.identity_path_b.trend import wait_for_owned_endpoint
from experiments.post_training.identity_path_b.trend_serve import event_identity, submission
from experiments.post_training.identity_path_b.workflow import ROOT_LIFETIME


def screen_worker(*, arm: str, output: str) -> None:
    """Use bundled target/prompt bytes; no child candidate-file transport or math."""
    if arm not in ("A0", "A3") or not output.startswith(TEMPORARY_PREFIX):
        raise ValueError("Require an authorizedA0/A3 target and freshRNO14d output")
    with payload() as (root, historical):
        event = CheckpointEvent(**json.loads((root / f"{arm}.json").read_bytes()))
        checked_export_event(event)
        destination = StoragePath(output)
        if destination.exists():
            raise FileExistsError("Scoped identity output already exists; no automatic retry")
        raw = event_bytes(event)
        event_uri = str(destination / "input-event.json")
        StoragePath(event_uri).write_bytes(raw)
        context = regional_task_context()
        assert context.client is not None
        credentials = {key: require_secret(key) for key in ("CW_KEY_ID", "CW_KEY_SECRET")}
        server = context.client.submit(**submission(event, event_uri, hashlib.sha256(raw).hexdigest(), credentials))
        status = {"arm": arm, "event": event.payload(), "serve_job": str(server.job_id), "status": "pending"}
        try:
            base_url = wait_for_owned_endpoint(server, f"/serve/{event_identity(event)}")

            def persist(index: int, record: dict):
                raw_record = (json.dumps(record, ensure_ascii=False) + "\n").encode()
                (destination / f"raw/{index:03d}.json").write_bytes(raw_record)

            records = generate(base_url, event.hf_export, root, historical, persist)
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
            # generate's executor drains all issued work before reaching this block.
            try:
                (destination / "status.json").write_bytes((json.dumps(status, indent=2) + "\n").encode())
            finally:
                server.cancel()


def screen_submission(*, name: str, arm: str, output: str, credentials: dict[str, str]) -> dict:
    """Build the scoped owner's native request without submitting."""
    if arm not in ("A0", "A3") or not name.startswith(f"pathb-identity-{arm.lower()}-r1-"):
        raise ValueError("Require the exact fresh authorized arm name")
    secrets = {key: credentials[key] for key in ("CW_KEY_ID", "CW_KEY_SECRET")}
    secrets.update(AWS_ACCESS_KEY_ID=secrets["CW_KEY_ID"], AWS_SECRET_ACCESS_KEY=secrets["CW_KEY_SECRET"])
    request = dict(
        name=name,
        entrypoint=Entrypoint.from_command(
            "bash",
            "-c",
            'exec "$IRIS_PYTHON" -u -m experiments.post_training.identity_path_b.identity_scope.owner '
            + shlex.join(["--arm", arm, "--output", output]),
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
    )
    request["priority_band"] = job_pb2.PRIORITY_BAND_INTERACTIVE
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A0", "A3"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.inspect_only:
        with payload() as (root, _):
            event = json.loads((root / f"{args.arm}.json").read_bytes())
            print(json.dumps({"arm": args.arm, "model": event["hf_export"], "output": args.output, "calls": 0}))
        return
    screen_worker(arm=args.arm, output=args.output)


if __name__ == "__main__":
    main()
