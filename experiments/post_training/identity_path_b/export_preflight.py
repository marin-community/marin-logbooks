# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One tiny CPU-owner storage write/read before each export GPU submission."""

import hashlib

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent, event_bytes


def export_write_probe(event: CheckpointEvent) -> dict:
    event.payload()
    raw = hashlib.sha256(event_bytes(event)).digest()
    path = StoragePath(event.eval_scratch) / "export-write-probe.bin"
    if path.exists():
        raise FileExistsError("Export preflight destination already exists; no automatic retry")
    path.write_bytes(raw)
    if path.read_bytes() != raw:
        raise ValueError("Export preflight tiny write/read differs")
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "application_attempts": 1}
