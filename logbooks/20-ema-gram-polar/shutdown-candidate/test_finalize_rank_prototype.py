# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fake-coordinator lifecycle tests plus real local Python atexit exit codes."""

import atexit
import json
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from finalize_rank_prototype import FinalizationFailure, exit_hook, finalize_rank
from test_rank_finalization_prototype import FakeCoordinator


class FinalizerChecks(unittest.TestCase):
    def run_ranks(self, failure=None):
        c = FakeCoordinator(4)
        shutdowns = []

        def run(rank):
            def drain():
                if rank == 0 and failure == "drain_exception":
                    raise ConnectionError("fake sink failure")
                return {
                    "required_complete": not (rank == 0 and failure == "required_incomplete"),
                    "auxiliary_complete": False,
                }

            def shutdown():
                self.assertEqual(len(c.key_value_dir_get("finalization-prototype/local/")), 4)
                shutdowns.append(rank)
                if rank == 1 and failure == "shutdown_exception":
                    raise RuntimeError("fake distributed shutdown failure")

            return finalize_rank(
                c,
                attempt_key="local",
                rank=rank,
                ranks=(0, 1, 2, 3),
                drain=drain,
                shutdown=shutdown,
                health_timeout_ms=5000,
            )

        results = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run, i) for i in range(4)]
            for f in futures:
                try:
                    results.append(f.result(timeout=5))
                except Exception as exc:
                    results.append(exc)
        self.assertEqual(set(shutdowns), set(range(4)))
        return results

    def test_auxiliary_failure_does_not_change_required_policy(self):
        for result in self.run_ranks():
            self.assertIsInstance(result, dict)
            self.assertTrue(all(r["required_complete"] for r in result.values()))
            self.assertTrue(all(not r["auxiliary_complete"] for r in result.values()))

    def test_required_failure_reaches_all_ranks_after_shutdown(self):
        self.assertTrue(all(isinstance(x, FinalizationFailure) for x in self.run_ranks("required_incomplete")))

    def test_drain_exception_reaches_all_ranks_after_shutdown(self):
        self.assertTrue(all(isinstance(x, FinalizationFailure) for x in self.run_ranks("drain_exception")))

    def test_shutdown_exception_not_hidden_by_other_rank_success(self):
        results = self.run_ranks("shutdown_exception")
        self.assertIsInstance(results[1], RuntimeError)
        # Other ranks may exit normally; the candidate deliberately has no
        # force-complete call which could override this task failure.
        self.assertEqual(sum(isinstance(x, dict) for x in results), 3)

    def test_actual_atexit_failure_status(self):
        p = subprocess.run([sys.executable, __file__, "--child", "failure"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)

    def test_actual_atexit_success_status(self):
        p = subprocess.run([sys.executable, __file__, "--child", "success"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":

        def finalizer():
            if sys.argv[2] == "failure":
                raise FinalizationFailure("injected finalizer failure")

        atexit.register(exit_hook, finalizer)
    else:
        r = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FinalizerChecks))
        Path(__file__).with_name("finalizer-prototype-checks.json").write_text(
            json.dumps(
                {
                    "passed": r.wasSuccessful(),
                    "tests": r.testsRun,
                    "production_integrated": False,
                    "runtime_qualified": False,
                    "scope": (
                        "fake coordinator and sinks; actual local Python atexit exit status; "
                        "no numerical kernels or remote jobs"
                    ),
                    "remaining": [
                        "Actual all-rank JAX shutdown health/lifetime behavior",
                        "Integration replacing the existing atexit hook after all wrapper JAX uses",
                        "Required W&B receipt production and optional Finelog receipt aggregation",
                        "Iris controller/runtime qualification without forced complete_job",
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        raise SystemExit(0 if r.wasSuccessful() else 1)
