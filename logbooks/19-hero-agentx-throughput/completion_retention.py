# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Save a finished child attempt before its enclosing GPU allocation ends."""

import hashlib
import json
import tempfile
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.agentx.artifacts import retain_artifacts, write_record


def retain_finished_attempt(output: Path, prefix: str, task_index: int, attempt_uid: str) -> dict:
    attempt = json.loads((output / "point/attempt.json").read_text())
    if attempt.get("status") not in ("succeeded", "failed") or not attempt.get("finished_unix"):
        raise ValueError("Cannot retain an unfinished attempt")
    if attempt.get("iris_attempt_uid") != attempt_uid:
        raise ValueError("Attempt identity differs from the current native task")
    destination = prefix_join(prefix, f"host-{task_index}/{attempt_uid}")
    with tempfile.TemporaryDirectory(prefix="hero-completed-output-", dir="/tmp") as temporary:
        bundle = Path(temporary) / "outputs.tar.zst"
        uri, digest = retain_artifacts(output, bundle, destination, name="outputs")
        filesystem, path = filesystem_for(uri)
        with filesystem.open(path, "rb") as stream:
            readback = hashlib.file_digest(stream, "sha256").hexdigest()
        if readback != digest:
            raise ValueError("Retained output failed SHA256 readback")
        receipt = {
            "task_id": attempt["iris_task_id"],
            "attempt_uid": attempt_uid,
            "host": task_index,
            "application_status": attempt["status"],
            "archive_uri": uri,
            "archive_sha256": digest,
            "readback_sha256": readback,
            "archive_bytes": bundle.stat().st_size,
            "scope": (
                "Finished host output only. Full-hour validity and numerical qualification require separate checks."
            ),
        }
        write_record(prefix_join(destination, "retention.json"), receipt)
    return receipt
