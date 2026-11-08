# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure Python local fault checks; no tracking services or numerical kernels."""

import dataclasses
import importlib.util
import json
import logging
import sys
import threading
import types
import unittest
from pathlib import Path
from typing import ClassVar

stub = types.ModuleType("levanter.tracker.tracker")
stub.Tracker = type("Tracker", (), {})
sys.modules["levanter.tracker.tracker"] = stub
path = Path(__file__).with_name("background_receipt_prototype.py")
spec = importlib.util.spec_from_file_location("receipt_prototype", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
logging.disable(logging.CRITICAL)


class Sink:
    name = "fake-external-sink"

    def __init__(self, fault=None, block=None):
        self.fault = fault
        self.block = block
        self.entered = threading.Event()
        self.release = threading.Event()
        self.delivered = []
        self.finishes = 0

    def _prepare_log(self, value):
        if self.fault == "prepare":
            raise ValueError("injected host materialization failure")
        return value

    def log(self, value, **kwargs):
        self._operation("log")
        self.delivered.append(value)

    def finish(self):
        self._operation("finish")
        self.finishes += 1

    def _operation(self, operation):
        if self.block == operation:
            self.entered.set()
            if not self.release.wait(5):
                raise RuntimeError("test failed to release sink")
        if self.fault == operation:
            raise RuntimeError("injected sink failure")

    def validate_artifact(self, *args, **kwargs):
        pass


class ReceiptChecks(unittest.TestCase):
    records: ClassVar[list[dict]] = []

    def tracker(self, sink, **kwargs):
        tracker = module.BackgroundTracker(sink, **kwargs)

        def cleanup():
            sink.release.set()
            # Cleanup only: release a worker for which the original best-effort
            # finish failed to enqueue a sentinel. No behavior assertion relies
            # on these private implementation details.
            if tracker._thread.is_alive():
                tracker._queue.put(module._SHUTDOWN, timeout=5)
                tracker._thread.join(5)

        self.addCleanup(cleanup)
        return tracker

    def record(self, tracker):
        receipt = tracker.finalization_receipt()
        self.records.append(
            {"case": self._testMethodName, **dataclasses.asdict(receipt), "locally_drained": receipt.locally_drained}
        )
        return receipt

    def test_success_and_idempotence(self):
        sink = Sink()
        t = self.tracker(sink, finish_timeout=5)
        t.log({"loss": 1}, step=0)
        t.finish()
        t.finish()
        self.assertEqual(sink.delivered, [{"loss": 1}])
        self.assertEqual(sink.finishes, 1)
        self.assertTrue(self.record(t).locally_drained)

    def test_finish_exception_observable(self):
        t = self.tracker(Sink(fault="finish"), finish_timeout=5)
        t.finish()
        r = self.record(t)
        self.assertEqual(r.finish_state, module.FinishState.FAILED)
        self.assertEqual(r.failed_calls, 1)
        self.assertFalse(r.locally_drained)

    def test_log_failure_not_erased_by_successful_finish(self):
        t = self.tracker(Sink(fault="log"), finish_timeout=5)
        t.log({"loss": 1}, step=0)
        t.finish()
        r = self.record(t)
        self.assertEqual(r.finish_state, module.FinishState.SUCCEEDED)
        self.assertEqual(r.failed_calls, 1)
        self.assertFalse(r.locally_drained)

    def test_prepare_failure_observable(self):
        t = self.tracker(Sink(fault="prepare"), finish_timeout=5)
        t.log({"loss": 1}, step=0)
        t.finish()
        r = self.record(t)
        self.assertEqual(r.failed_calls, 1)
        self.assertFalse(r.locally_drained)

    def test_blocked_finish_keeps_immutable_incomplete_receipt(self):
        sink = Sink(block="finish")
        t = self.tracker(sink, finish_timeout=0)
        t.finish()
        self.assertTrue(sink.entered.wait(5))
        r = self.record(t)
        self.assertTrue(r.worker_alive)
        self.assertFalse(r.locally_drained)
        sink.release.set()
        self.assertTrue(t._wait_until_idle(5))
        t.finish()
        self.assertEqual(sink.finishes, 1)
        self.assertFalse(r.locally_drained)

    def test_full_queue_preserves_drops_and_missing_finish(self):
        sink = Sink(block="log")
        t = self.tracker(sink, max_queue_size=1, finish_timeout=0)
        t.log({"label": 0}, step=0)
        self.assertTrue(sink.entered.wait(5))
        t.log({"label": 1}, step=1)
        t.log({"label": 2}, step=2)
        t.finish()
        r = self.record(t)
        self.assertEqual(r.finish_state, module.FinishState.NOT_ENQUEUED)
        self.assertEqual(r.rejected_updates, 1)
        self.assertFalse(r.locally_drained)
        self.assertEqual(sink.finishes, 0)

    def test_missing_artifact_rejects_clean_receipt(self):
        t = self.tracker(Sink(), finish_timeout=5)
        t.log_artifact(Path(__file__).with_name("nonexistent-artifact-for-test"))
        t.finish()
        r = self.record(t)
        self.assertEqual(r.rejected_updates, 1)
        self.assertFalse(r.locally_drained)

    def test_post_finish_update_is_visible(self):
        t = self.tracker(Sink(), finish_timeout=5)
        t.finish()
        t.log({"label": 1}, step=1)
        r = self.record(t)
        self.assertEqual(r.rejected_updates, 1)
        self.assertFalse(r.locally_drained)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReceiptChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "passed": result.wasSuccessful(),
        "tests": result.testsRun,
        "cases": ReceiptChecks.records,
        "scope": "isolated prototype; fake sink, stub abstract Tracker base; no JAX/W&B imports or network",
        "runtime_qualified": False,
        "production_integrated": False,
    }
    Path(__file__).with_name("receipt-prototype-checks.json").write_text(json.dumps(report, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
