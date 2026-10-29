# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact checkpoint weights with the frozen native T2 serving engine/profile."""

import argparse
import hashlib
import json
import shlex
from dataclasses import replace
from pathlib import Path

from iris.cluster.types import Entrypoint
from marin.inference.iris import run_iris_service
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.post_training.identity_path_b.async_launch import CHILD_LIFETIME, checked_export_event
from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent, event_bytes
from experiments.post_training.identity_path_b.replay_serve import T2_SERVE_SHA256
from experiments.post_training.identity_t2 import serve as t2


def event_identity(event: CheckpointEvent) -> str:
    return f"pathb-trend-{event.arm}-{event.completed_updates:06d}-{hashlib.sha256(event_bytes(event)).hexdigest()[:12]}"


def service_config(event: CheckpointEvent):
    if hashlib.sha256(Path(t2.__file__).read_bytes()).hexdigest() != T2_SERVE_SHA256:
        raise ValueError("The frozen native T2 serve source changed")
    original = t2.service_config()
    return replace(
        original,
        model=replace(original.model, weights=event.hf_export),
        endpoint_name=f"/serve/{event_identity(event)}",
        timeout_hours=12,
        controller_proxy_timeout_seconds=1800,
    )


def submission(event: CheckpointEvent, event_uri: str, event_sha256: str, credentials: dict[str, str]) -> dict:
    kwargs = t2.submission(credentials)
    script = (
        'set -euo pipefail\nroot_python="$IRIS_VENV/bin/python"\n'
        "source /app/vendor/skyrl/.iris-runtime-env\n"
        'exec "$root_python" -u -m experiments.post_training.identity_path_b.trend_serve '
        f"--event-uri {shlex.quote(event_uri)} --event-sha256 {shlex.quote(event_sha256)}"
    )
    kwargs.update(
        name=event_identity(event),
        entrypoint=Entrypoint(command=["bash", "-c", script]),
        timeout=Duration.from_seconds(CHILD_LIFETIME),
        # Already inside the RNO peer task, matching OPD's ambient route.
        constraints=None,
    )
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-uri", required=True)
    parser.add_argument("--event-sha256", required=True)
    args = parser.parse_args()
    raw = StoragePath(args.event_uri).read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.event_sha256:
        raise ValueError("Trend serving input event hash changed")
    payload = json.loads(raw)
    event = CheckpointEvent(**{k: v for k, v in payload.items() if k in CheckpointEvent.__dataclass_fields__})
    checked_export_event(event)
    t2.validate_runtime()
    t2.configure_worker_s3()
    run_iris_service(service_config(event))


if __name__ == "__main__":
    main()
