# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Committed checkpoint events for independent in-region export and small evals."""

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointEvent:
    completed_updates: int
    native_checkpoint: str
    hf_export: str
    exposure: dict
    training_pin: str
    packed_data_sha256: str
    arm: str

    def payload(self) -> dict:
        if not self.native_checkpoint.startswith("s3://marin-rno2a/") or not self.hf_export.startswith(
            "s3://marin-rno2a/"
        ):
            raise ValueError("Checkpoint and export must remain in the training region")
        return {
            "schema_version": 1,
            "arm": self.arm,
            "completed_updates": self.completed_updates,
            "native_checkpoint": self.native_checkpoint,
            "hf_export": self.hf_export,
            "exposure": self.exposure,
            "training_pin": self.training_pin,
            "packed_data_sha256": self.packed_data_sha256,
            "screen": {
                "identity52": "off; trained-profile and Snowball scores",
                "third_person": "dev14",
                "gsm8k": "one decode",
                "bfcl": "optional, executor decision; no capacity census",
            },
            "final_panel": "held out; no panel60/unseen30/full paired math in this trend job",
        }


def event_bytes(event: CheckpointEvent) -> bytes:
    return (json.dumps(event.payload(), sort_keys=True, separators=(",", ":")) + "\n").encode()


class AsyncCheckpoints:
    """Persist committed snapshots, then submit separate export/eval work asynchronously.

    The callback submits jobs and returns their identifiers; it must never wait
    for evaluations. There are zero submission retries. The immutable event is
    retained even if submission fails, so a failed trend cannot disappear.
    """

    def __init__(
        self,
        *,
        write_event: Callable[[str, bytes], None],
        submit_export_eval: Callable[[dict], dict],
        write_submission_result: Callable[[str, dict], None],
    ):
        self.write_event = write_event
        self.submit_export_eval = submit_export_eval
        self.write_submission_result = write_submission_result
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="path-b-submit")
        self.pending = []

    def publish_committed(self, event: CheckpointEvent) -> str:
        """Call on the writer only AFTER the native checkpoint is committed."""
        raw = event_bytes(event)
        digest = hashlib.sha256(raw).hexdigest()
        name = f"{event.arm}-updates-{event.completed_updates:06d}-{digest[:12]}"
        self.write_event(name, raw)
        self.pending.append(self.executor.submit(self.submit, name, event.payload()))
        return name

    def submit(self, name: str, payload: dict) -> None:
        try:
            result = {"submitted": True, "jobs": self.submit_export_eval(payload)}
        except Exception as error:
            result = {"submitted": False, "error_type": type(error).__name__, "retry_count": 0}
        self.write_submission_result(name, result)

    def finish_submissions(self) -> None:
        """Wait only for bounded submit calls, never for export/evaluation completion."""
        self.executor.shutdown(wait=True)
        for pending in self.pending:
            pending.result()
