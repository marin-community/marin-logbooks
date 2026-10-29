# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two live native snapshots; retire only an exact, audited current-run event."""

import json
import time

import numpy as np
from jax.experimental import multihost_utils
from levanter.compat.hf_export import run_on_export_writer
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Deadline

from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent, event_bytes

BOUNDARY_WAIT_SECONDS = 2 * 3600
MAX_NATIVE_CHECKPOINTS = 2


def await_checkpoint_slot(paths: list[str], *, final: bool) -> bool:
    """Every training rank polls its own published paths, without a pending collective.

    Nonfinal boundaries skip after two hours; final has no algorithmic deadline.
    Native task lifetimes remain finite. Failed/queued exports keep their slots;
    they are never pruned, repaired, or automatically retried.
    """
    deadline = None if final else Deadline.from_seconds(BOUNDARY_WAIT_SECONDS)
    while True:

        def check():
            if sum(StoragePath(path).exists() for path in paths) < MAX_NATIVE_CHECKPOINTS:
                return 1
            return -1 if deadline is not None and deadline.expired() else 0

        # Short matching collectives each poll, never a two-hour leader wait
        # inside a pending collective. All ranks make the SAME save/skip choice.
        status = run_on_export_writer(check)
        status = int(multihost_utils.broadcast_one_to_all(np.asarray(status or 0, dtype=np.int32)))
        if status:
            return status == 1
        time.sleep(2)


def delete_audited_checkpoint(event: CheckpointEvent, run_root: str, source_sha256: str) -> None:
    """CPU owner only, after the export task is terminal and its audit is complete."""
    expected = f"{run_root}/checkpoints/step-{event.completed_updates - 1}"
    if event.native_checkpoint != expected or not event.hf_export.startswith(f"{run_root}/hf/"):
        raise ValueError("Checkpoint retirement must target this exact current-run step")
    if "/r5-" not in run_root:
        raise ValueError("Retirement is restricted to fresh r5 output namespaces")
    stored = json.loads((StoragePath(event.hf_export) / "path-b-checkpoint-event.json").read_bytes())
    if stored != event.payload():
        raise ValueError("Cannot retire native without the exact audited export completion marker")
    audit = json.loads((StoragePath(event.hf_export) / "path-b-storage-audit.json").read_bytes())
    tensors = audit["tensors"]
    if (
        audit["schema"] != "path_b_storage_v1"
        or audit["source_receipt_sha256"] != source_sha256
        or audit["frozen_native_tensors"] != 104
        or audit["trained_master_cast_tensors"] != 398
        or len(tensors) != 502
        or sum(row["frozen_source"] for row in tensors.values()) != 104
    ):
        raise ValueError("Native retirement requires the full502 storage-aware export audit")
    checkpoint = StoragePath(expected)
    metadata = json.loads((checkpoint / "metadata.json").read_bytes())
    if metadata["step"] != event.completed_updates - 1 or metadata["is_temporary"] is not True:
        raise ValueError("Native retirement requires matching committed temporary checkpoint metadata")
    checkpoint.rmtree()
    if checkpoint.exists():
        raise RuntimeError("Audited native deletion did not complete")

    # Small event/receipt survives removal of the large native tree. If writing
    # this fails, the workflow records that explicit diagnostic failure.
    (StoragePath(run_root) / f"checkpoint-retirement/update-{event.completed_updates}.json").write_bytes(
        event_bytes(event)
    )


def save_temporary_checkpoint(checkpointer, tree, *, step: int) -> None:
    """Use native temporary serialization without its save-order prune callback.

    Out-of-order audit retirement makes native historical-list pruning unsafe.
    The physical-slot check before this call and audit-driven retirement are the
    sole retention authority. No default on_step hooks may run for these workers.
    """
    checkpointer.save_checkpoint(tree=tree, step=step, destination=f"step-{step}", is_temporary=True)
