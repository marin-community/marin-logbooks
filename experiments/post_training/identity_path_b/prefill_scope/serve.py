# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The frozen T2 A3 service with one explicit native template override."""

import argparse
import hashlib
import json
from dataclasses import replace

from marin.inference.iris import run_iris_service
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.identity_path_b import trend_serve
from experiments.post_training.identity_path_b.async_launch import checked_export_event
from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent
from experiments.post_training.identity_path_b.prefill_scope.data import template


def service_config(event: CheckpointEvent):
    original = trend_serve.service_config(event)
    if event.arm != "A3":
        raise ValueError("The prefill-only screen is authorized for A3 only")
    return replace(original, model=replace(original.model, chat_template_content=template()))


def submission(event: CheckpointEvent, event_uri: str, event_sha256: str, credentials: dict) -> dict:
    request = trend_serve.submission(event, event_uri, event_sha256, credentials)
    old = "experiments.post_training.identity_path_b.trend_serve "
    command = request["entrypoint"].command[-1]
    if command.count(old) != 1:
        raise ValueError("Unexpected native serve entrypoint")
    request["entrypoint"].command[-1] = command.replace(
        old, "experiments.post_training.identity_path_b.prefill_scope.serve "
    )
    return request


def validate_export_template(hf_export: str) -> None:
    expected = {
        "tokenizer_config.json": "d9270d9407a15ec895fa2cdb0b6fbe9cd7dab937b91c2db708a918b35efa18b3",
        "chat_template.jinja": "b3f20bc21498407f2815dd1a6c26150192926ccf04f6324ca40a1209bff7d1de",
    }
    for name, digest in expected.items():
        if hashlib.sha256((StoragePath(hf_export) / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Actual A3 export tokenizer/template metadata changed: " + name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-uri", required=True)
    parser.add_argument("--event-sha256", required=True)
    args = parser.parse_args()
    raw = StoragePath(args.event_uri).read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.event_sha256:
        raise ValueError("Prefill serve event changed")
    payload = json.loads(raw)
    event = CheckpointEvent(
        **{key: value for key, value in payload.items() if key in CheckpointEvent.__dataclass_fields__}
    )
    checked_export_event(event)
    validate_export_template(event.hf_export)
    trend_serve.t2.validate_runtime()
    trend_serve.t2.configure_worker_s3()
    run_iris_service(service_config(event))


if __name__ == "__main__":
    main()
