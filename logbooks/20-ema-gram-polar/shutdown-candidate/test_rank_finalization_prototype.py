# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local concurrency/fault checks of the proposed coordinator exchange only."""

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rank_finalization_prototype import exchange_finalization_receipts


class FakeCoordinator:
    def __init__(self, peers):
        self.values = {}
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(peers)
        self.first_three_waiting = threading.Event()
        self.arrivals = 0

    def key_value_set(self, key, value, allow_overwrite=False):
        with self.lock:
            if key in self.values and not allow_overwrite:
                raise RuntimeError("duplicate execution key")
            self.values[key] = value

    def key_value_dir_get(self, prefix):
        with self.lock:
            return [(k, v) for k, v in self.values.items() if k.startswith(prefix)]

    def wait_at_barrier(self, barrier_id, timeout_in_ms, process_ids):
        with self.lock:
            self.arrivals += 1
            if self.arrivals == 3:
                self.first_three_waiting.set()
        self.barrier.wait(timeout=timeout_in_ms / 1000)


class RankChecks(unittest.TestCase):
    def exchange(self, client, rank, receipt, timeout=5000, attempt="local-attempt"):
        return exchange_finalization_receipts(
            client, attempt_key=attempt, rank=rank, ranks=(0, 1, 2, 3), local_receipt=receipt, health_timeout_ms=timeout
        )

    def test_fast_ranks_wait_for_slow_rank_receipt(self):
        c = FakeCoordinator(4)
        with ThreadPoolExecutor(max_workers=4) as pool:
            fast = [pool.submit(self.exchange, c, i, {"complete": True}) for i in (1, 2, 3)]
            self.assertTrue(c.first_three_waiting.wait(5))
            self.assertTrue(all(not f.done() for f in fast))
            slow = pool.submit(self.exchange, c, 0, {"complete": True})
            for f in [*fast, slow]:
                self.assertEqual(f.result(timeout=5), {i: {"complete": True} for i in range(4)})

    def test_failed_receipt_visible_to_every_rank(self):
        c = FakeCoordinator(4)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(self.exchange, c, i, {"complete": i != 0, "reason": "drain_failed" if i == 0 else None})
                for i in range(4)
            ]
            for f in futures:
                receipts = f.result(timeout=5)
                self.assertFalse(receipts[0]["complete"])
                self.assertEqual(receipts[0]["reason"], "drain_failed")
                self.assertEqual(len(receipts), 4)

    def test_missing_rank_does_not_release_success(self):
        c = FakeCoordinator(4)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(self.exchange, c, i, {"complete": True}, 50) for i in (1, 2, 3)]
            for f in futures:
                with self.assertRaises(threading.BrokenBarrierError):
                    f.result(timeout=5)

    def test_coordinator_error_remains_failure(self):
        class FailedCoordinator(FakeCoordinator):
            def wait_at_barrier(self, *args):
                raise ConnectionError("injected coordinator failure")

        with self.assertRaises(ConnectionError):
            self.exchange(FailedCoordinator(4), 0, {"complete": True})

    def test_stale_extra_rank_rejected(self):
        c = FakeCoordinator(4)
        c.key_value_set("finalization-prototype/local-attempt/99", '{"complete":true}')
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(self.exchange, c, i, {"complete": True}) for i in range(4)]
            for f in futures:
                with self.assertRaises(RuntimeError):
                    f.result(timeout=5)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RankChecks))
    Path(__file__).with_name("rank-prototype-checks.json").write_text(
        json.dumps(
            {
                "passed": result.wasSuccessful(),
                "tests": result.testsRun,
                "scope": "pure Python fake coordinator; no JAX initialization, numerical kernels, remote reads or jobs",
                "proves": (
                    "Readiness-exchange ordering and preservation of failure receipts "
                    "under the fake coordinator contract."
                ),
                "does_not_prove": [
                    "Actual coordinator heartbeat/lifetime semantics",
                    "Safe JAX shutdown or atexit ordering",
                    "Remote telemetry delivery",
                    "Canonical success gating after shutdown",
                ],
                "runtime_qualified": False,
                "production_integrated": False,
            },
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)
