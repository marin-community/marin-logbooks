# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actual candidate installer/core, fake services, real local trace persistence."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from trace_contract import Identity, Trace, audit_clean_trace

SOURCE = (
    Path(os.environ["EMA_GRAM_CANDIDATE_ROOT"])
    / "lib/levanter/tests/shutdown_candidate_checks/check_hook_integration.py"
)
spec = importlib.util.spec_from_file_location("hook_fixture", SOURCE)
hooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hooks)


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Fixture labels do not attest to the installed source revision.
        candidate = "LOCAL_SOURCE_UNATTESTED"
        self.manifest = {
            "campaign": "fixture",
            "scenario": "clean",
            "candidate_commit": candidate,
            "execution_id": "fixture-world",
            "ranks": [0, 1, 2, 3],
            "receipt_prefix": "levanter/finalization/v1/local-job/",
            "require_hook_lifecycle": True,
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
        for rank in range(4):
            path = Path(self.temp.name) / f"{rank}.jsonl"
            self.paths.append(path)
            trace = Trace(
                path,
                Identity(
                    "fixture",
                    "clean",
                    candidate,
                    "fixture-world",
                    f"attempt-{rank}",
                    f"task-{rank}",
                    f"worker-{rank}",
                    rank,
                    (0, 1, 2, 3),
                    f"fake-online-{rank}",
                ),
            )
            handlers = hooks.Hooks()
            ns, effects = hooks.load(handlers, hooks.Coordinator(), rank=rank)
            ns["install_coordinated_iris_finalizer"](
                hooks.good(),
                application_error=None,
                health_timeout_ms=120000,
                observation=hooks.core.FinalizationObservation(trace.observe),
            )
            handlers.run()
            trace.close()
            self.assertEqual(effects, ["shutdown"])

    def rewrite(self, mutate):
        path = self.paths[0]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        mutate(rows)
        for i, row in enumerate(rows):
            row["seq"] = i
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_installed_hook_trace_audits_without_global_client_proxy(self):
        verdict = audit_clean_trace(self.paths, self.manifest)
        self.assertTrue(verdict["local_clean_ordering_pass"], verdict["errors"])
        self.assertFalse(verdict["runtime_qualified"])

    def test_missing_hook_return_cannot_certify_lifecycle(self):
        self.rewrite(lambda rows: rows.pop())
        self.assertFalse(audit_clean_trace(self.paths, self.manifest)["local_clean_ordering_pass"])

    def test_direct_barrier_from_other_world_is_rejected(self):
        def mutate(rows):
            next(row for row in rows if row["event"] == "read_return")["fields"]["key"] = "foreign/read"

        self.rewrite(mutate)
        self.assertFalse(audit_clean_trace(self.paths, self.manifest)["local_clean_ordering_pass"])


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    (Path(os.environ["EMA_GRAM_CHECK_RESULT_DIR"]) / "observed-lifecycle-checks.json").write_text(
        json.dumps(
            {
                "passed": result.wasSuccessful(),
                "tests": result.testsRun,
                "scope": (
                    "Actual installer AST and core plus local persisted traces; "
                    "four independent fake rank fixtures, not a shared distributed service."
                ),
                "runtime_qualified": False,
                "new_jobs": 0,
                "numerical_kernels": 0,
            },
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)
