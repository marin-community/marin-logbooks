# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic trace fixtures only; no actual runtime qualification evidence."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from trace_contract import Identity, Trace, TracedCoordinator, audit_clean_trace, traced_call, traced_shutdown

SOURCE = Path(os.environ["EMA_GRAM_CANDIDATE_ROOT"]) / "lib/levanter/src/levanter/distributed_finalization.py"
spec = importlib.util.spec_from_file_location("core", SOURCE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
COMMIT = "LOCAL_SOURCE_UNATTESTED"


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.prefix = "levanter/finalization/v1/fixture/"
        self.manifest = {
            "campaign": "fixture",
            "scenario": "clean",
            "candidate_commit": COMMIT,
            "execution_id": "world-fixture",
            "ranks": [0, 1, 2, 3],
            "receipt_prefix": self.prefix,
            "rank_bindings": {
                str(r): {
                    "task_uid": f"task-{r}",
                    "attempt_uid": f"attempt-{r}",
                    "worker_id": f"worker-{r}",
                    "online_run_id": f"fake-online-{r}",
                }
                for r in range(4)
            },
        }
        self.paths = []
        self.calls = {}
        for rank in range(4):
            path = Path(self.temp.name) / f"{rank}.jsonl"
            self.paths.append(path)
            identity = Identity(
                "fixture",
                "clean",
                COMMIT,
                "world-fixture",
                f"attempt-{rank}",
                f"task-{rank}",
                f"worker-{rank}",
                rank,
                (0, 1, 2, 3),
                f"fake-online-{rank}",
            )
            trace = Trace(path, identity)
            calls = []
            self.calls[rank] = calls
            prefix = self.prefix

            class FakeCoordinator:
                def __init__(self, call_log, key_prefix):
                    self.calls = call_log
                    self.prefix = key_prefix

                def key_value_set(self, key, value, allow_overwrite=False):
                    self.calls.append("set")

                def wait_at_barrier(self, key, timeout, processes):
                    self.calls.append(key.rsplit("/", 1)[-1])

                def key_value_dir_get(self, key):
                    self.calls.append("get")
                    return [
                        (self.prefix + str(r), json.dumps({"required_complete": True, "application_failed": False}))
                        for r in range(4)
                    ]

            core.finalize_and_shutdown(
                TracedCoordinator(FakeCoordinator(calls, prefix), trace),
                world_key="fixture",
                rank=rank,
                ranks=(0, 1, 2, 3),
                required_complete=True,
                application_failed=False,
                health_timeout_ms=1000,
                shutdown=traced_shutdown(lambda calls=calls: calls.append("shutdown"), trace),
            )
            trace.close()

    def mutate(self, index, edit):
        path = self.paths[index]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        edit(rows)
        for i, row in enumerate(rows):
            row["seq"] = i
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_actual_core_trace_and_no_extra_coordinator_calls(self):
        result = audit_clean_trace(self.paths, self.manifest)
        self.assertTrue(result["local_clean_ordering_pass"])
        self.assertFalse(result["runtime_qualified"])
        self.assertTrue(all(calls == ["set", "ready", "get", "read", "shutdown"] for calls in self.calls.values()))

    def test_shutdown_before_read_ack_rejected(self):
        def reorder(rows):
            index = next(i for i, r in enumerate(rows) if r["event"] == "shutdown_enter")
            row = rows.pop(index)
            target = next(
                i for i, r in enumerate(rows) if r["event"] == "barrier_return" and r["fields"]["phase"] == "read"
            )
            rows.insert(target, row)
            for i, r in enumerate(rows):
                r["monotonic_ns"] = i

        self.mutate(0, reorder)
        self.assertFalse(audit_clean_trace(self.paths, self.manifest)["local_clean_ordering_pass"])

    def test_foreign_attempt_rejected(self):
        def change(rows):
            for r in rows:
                r["identity"]["attempt_uid"] = "foreign-attempt"

        self.mutate(1, change)
        self.assertFalse(audit_clean_trace(self.paths, self.manifest)["local_clean_ordering_pass"])

    def test_missing_host_rejected(self):
        self.assertFalse(audit_clean_trace(self.paths[:3], self.manifest)["local_clean_ordering_pass"])

    def test_foreign_receipt_key_rejected(self):
        def change(rows):
            row = next(r for r in rows if r["event"] == "receipt_read_return")
            row["fields"]["keys"][0] = "foreign/0"

        self.mutate(2, change)
        self.assertFalse(audit_clean_trace(self.paths, self.manifest)["local_clean_ordering_pass"])

    def test_callback_error_recorded_and_original_preserved(self):
        path = Path(self.temp.name) / "failed.jsonl"
        trace = Trace(
            path, Identity("fixture", "missing-peer", COMMIT, "world", "attempt", "task", "worker", 0, (0,), "fake-run")
        )
        original = TimeoutError("injected communication failure")

        def fail():
            raise original

        with self.assertRaises(TimeoutError) as caught:
            traced_call(trace, "barrier", fail)
        trace.close()
        self.assertIs(caught.exception, original)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(rows[-1]["event"], "barrier_error")
        self.assertEqual(rows[-1]["fields"], {"error_type": "TimeoutError"})

    def test_error_logging_failure_does_not_replace_original_exit(self):
        class BrokenErrorTrace:
            def emit(self, event, **fields):
                if event.endswith("_error"):
                    raise OSError("injected trace storage error")

        original = SystemExit(9)

        def fail():
            raise original

        with self.assertRaises(SystemExit) as caught:
            traced_shutdown(fail, BrokenErrorTrace())()
        self.assertIs(caught.exception, original)
        self.assertEqual(caught.exception.code, 9)
        self.assertIn("OSError", original.__notes__[0])

    def test_failed_operation_cannot_be_hidden_among_clean_events(self):
        def insert(rows):
            extra = dict(rows[-1])
            extra["event"] = "barrier_error"
            extra["fields"] = {"error_type": "TimeoutError"}
            rows.insert(-1, extra)

        self.mutate(0, insert)
        self.assertFalse(audit_clean_trace(self.paths, self.manifest)["local_clean_ordering_pass"])

    def test_truncated_trace_is_not_clean(self):
        self.mutate(3, lambda rows: rows.pop())
        self.assertFalse(audit_clean_trace(self.paths, self.manifest)["local_clean_ordering_pass"])


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    (Path(os.environ["EMA_GRAM_CHECK_RESULT_DIR"]) / "trace-contract-checks.json").write_text(
        json.dumps(
            {
                "passed": result.wasSuccessful(),
                "tests": result.testsRun,
                "scope": (
                    "actual candidate core plus trace adapter; synthetic four-rank fixtures and fake coordinator; "
                    "no shared real distributed runtime"
                ),
                "runtime_qualified": False,
                "new_jobs": 0,
                "numerical_kernels": 0,
                "remote_calls": 0,
            },
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)
