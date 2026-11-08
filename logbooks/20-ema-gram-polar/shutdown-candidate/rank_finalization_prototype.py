# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Proposed host-only readiness exchange, not connected to JAX or any job.

Call after the rank's final JAX operation and attempted local telemetry drain.
This exchanges evidence; it neither shuts down JAX nor marks a job successful.
The caller still needs a reviewed failure policy and real runtime qualification.
"""

import json
from collections.abc import Sequence
from typing import Protocol


class Coordinator(Protocol):
    def key_value_set(self, key: str, value: str, allow_overwrite: bool = False) -> None: ...
    def key_value_dir_get(self, key: str) -> list[tuple[str, str]]: ...
    def wait_at_barrier(self, barrier_id: str, timeout_in_ms: int, process_ids: Sequence[int]) -> None: ...


def exchange_finalization_receipts(
    client: Coordinator,
    *,
    attempt_key: str,
    rank: int,
    ranks: tuple[int, ...],
    local_receipt: dict,
    health_timeout_ms: int,
) -> dict[int, dict]:
    """Wait for every rank's receipt using coordinator I/O, without JAX kernels.

    The attempt key must be unique to the execution attempt. No keys are deleted:
    a peer still reading must not lose evidence. Failed receipts are returned,
    never relabeled as success. Missing peers or malformed evidence raise.
    """
    if rank not in ranks or len(set(ranks)) != len(ranks):
        raise ValueError("invalid rank membership")
    prefix = f"finalization-prototype/{attempt_key}/"
    client.key_value_set(f"{prefix}{rank}", json.dumps(local_receipt, allow_nan=False))
    client.wait_at_barrier(f"{prefix}ready", health_timeout_ms, ranks)
    receipts = {int(key.removeprefix(prefix)): json.loads(value) for key, value in client.key_value_dir_get(prefix)}
    if set(receipts) != set(ranks):
        raise RuntimeError("finalization receipt membership does not match execution")
    return receipts
