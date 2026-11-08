# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolated future finalizer candidate; not installed into any execution path.

Invoke only after the outer wrapper's final JAX uses. A real integration must
replace (not supplement) the existing completion/shutdown hook. Successful
process exits are left to Iris task reconciliation; this code never calls the
force-complete-job API. Auxiliary diagnostic loss is recorded separately from
required W&B completion. No local receipt certifies remote history integrity.
"""

import logging
import os
from collections.abc import Callable

from rank_finalization_prototype import Coordinator, exchange_finalization_receipts


class FinalizationFailure(RuntimeError):
    pass


def finalize_rank(
    client: Coordinator,
    *,
    attempt_key: str,
    rank: int,
    ranks: tuple[int, ...],
    drain: Callable[[], dict],
    shutdown: Callable[[], None],
    health_timeout_ms: int,
) -> dict[int, dict]:
    """Exchange required-sink evidence, shut down together, preserve failures."""
    try:
        receipt = drain()
        if type(receipt.get("required_complete")) is not bool:
            raise ValueError("missing required-sink completion evidence")
    except Exception as exc:
        receipt = {"required_complete": False, "drain_exception_type": type(exc).__name__}
    receipts = exchange_finalization_receipts(
        client,
        attempt_key=attempt_key,
        rank=rank,
        ranks=ranks,
        local_receipt=receipt,
        health_timeout_ms=health_timeout_ms,
    )
    # All live ranks have finished their local drain attempt. Keep shutdown
    # coordinated even when one required sink failed; propagate failure after it.
    shutdown()
    if any(r.get("required_complete") is not True for r in receipts.values()):
        raise FinalizationFailure("required telemetry incomplete on one or more ranks")
    return receipts


def exit_hook(finalize: Callable[[], None]) -> None:
    """Unlike an uncaught atexit exception, preserve failure in process exit status."""
    try:
        finalize()
    except BaseException:
        logging.getLogger(__name__).exception("Finalization failed; preserving nonzero exit status")
        os._exit(1)
