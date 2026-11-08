# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run only draft orchestration with fake service/import boundaries."""

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cpu_entrypoint import BOUND_MODULES, DiagnosticSpec, run_diagnostic

CANDIDATE = Path(os.environ["EMA_GRAM_CANDIDATE_ROOT"]).resolve()
RESULT_DIR = Path(os.environ["EMA_GRAM_CHECK_RESULT_DIR"]).resolve()
if not RESULT_DIR.is_dir():
    raise ValueError("check result directory must already exist")


@dataclass
class Report:
    name: str = "wandb"
    state: str = "complete"


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for relative in BOUND_MODULES.values():
            file = self.root / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("source fixture")
        self.spec = DiagnosticSpec(
            "fixture",
            "clean",
            "synthetic-source",
            self.root,
            {
                relative: hashlib.sha256((self.root / relative).read_bytes()).hexdigest()
                for relative in BOUND_MODULES.values()
            },
            "fake-project",
            "fake-entity",
            120000,
        )
        self.env = patch.dict(
            os.environ,
            {
                "JAX_PLATFORMS": "cpu",
                "IRIS_TASK_ID": "synthetic-task",
                "IRIS_ATTEMPT_UID": "synthetic-attempt",
                "IRIS_WORKER_ID": "synthetic-worker",
            },
            clear=True,
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.events = []
        self.installs = []
        self.mode = "online"
        self.init_error = None
        self.finish_error = None
        self.initialized = False
        self.existing_run = None
        self.shadow_module = None
        self.worker_id = "synthetic-worker"
        self.rank = 0
        self.report_state = "complete"

    def importer(self, name):
        module = self.fake_import(name)
        if name in BOUND_MODULES:
            module.__file__ = (
                str(self.root / BOUND_MODULES[name]) if name != self.shadow_module else "/foreign/module.py"
            )
        return module

    def fake_import(self, name):
        self.events.append("import:" + name)
        if name == "wandb":

            def init(**kwargs):
                self.events.append("online-init")
                self.assertEqual(kwargs["mode"], "online")
                self.assertEqual(kwargs["resume"], "never")
                return SimpleNamespace(
                    id=kwargs["id"],
                    settings=SimpleNamespace(mode=self.mode),
                    url="fake://run",
                    config={},
                    finish=lambda: self.events.append("native-finish"),
                )

            return SimpleNamespace(init=init, run=self.existing_run)
        self.assertIn("online-init", self.events)
        if name == "levanter.tracker.wandb":
            return SimpleNamespace(WandbTracker=lambda **kwargs: object())
        if name == "levanter.tracker.background":
            outer = self

            class Tracker:
                def __init__(self, wrapped):
                    pass

                def log(self, metrics, **kwargs):
                    outer.events.append(("sentinel", metrics["diagnostic/sentinel"]))

                def finish_with_report(self):
                    outer.events.append("tracker-finish")
                    if outer.finish_error:
                        raise outer.finish_error
                    return Report(state=outer.report_state)

            return SimpleNamespace(BackgroundTracker=Tracker)
        if name == "levanter.distributed":

            def install(report, **kwargs):
                self.installs.append((report, kwargs))
                self.events.append("install")

            return SimpleNamespace(install_coordinated_iris_finalizer=install)
        if name == "levanter.distributed_finalization":
            return SimpleNamespace(FinalizationObservation=lambda sink: sink)
        if name == "iris.jax.init":

            def initialize():
                self.events.append("world-init")
                if self.init_error:
                    raise self.init_error
                self.initialized = True

            return SimpleNamespace(initialize_jax=initialize)
        if name == "levanter.tracker.finalization":
            return SimpleNamespace(required_sinks_complete=lambda report, required: report.state == "complete")
        if name == "levanter.tracker.tracker":
            return SimpleNamespace()
        if name == "iris.cluster.client.job_info":
            info = SimpleNamespace(
                task_attempt=SimpleNamespace(to_wire=lambda: "synthetic-task"),
                attempt_uid="synthetic-attempt",
                worker_id=self.worker_id,
                num_tasks=4,
                task_index=self.rank,
                attempt_id=0,
                job_id=SimpleNamespace(to_wire=lambda: "synthetic-job"),
            )
            return SimpleNamespace(get_job_info=lambda: info)
        if name == "jax":
            return SimpleNamespace(
                distributed=SimpleNamespace(is_initialized=lambda: self.initialized),
                process_index=lambda: self.rank,
                process_count=lambda: 4,
            )
        raise AssertionError(name)

    def invoke(self):
        run_diagnostic(self.spec, self.root / "output", importer=self.importer)

    def test_online_precedes_runtime_and_finish_precedes_hook(self):
        self.invoke()
        self.assertLess(self.events.index("online-init"), self.events.index("import:levanter.tracker.wandb"))
        self.assertLess(self.events.index("tracker-finish"), self.events.index("install"))
        self.assertEqual([e for e in self.events if isinstance(e, tuple)], [("sentinel", i) for i in range(4)])
        self.assertIsNone(self.installs[0][1]["application_error"])
        rows = [json.loads(x) for x in (self.root / "output/events.jsonl").read_text().splitlines()]
        self.assertEqual(rows[-1]["event"], "tracker_report")

    def test_offline_return_cannot_initialize_world(self):
        self.mode = "offline"
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertNotIn("world-init", self.events)
        self.assertFalse(any(x.startswith("import:levanter") for x in self.events if isinstance(x, str)))
        self.assertEqual(self.installs, [])

    def test_failed_world_initialization_finishes_tracker_without_install(self):
        error = OSError("fake init failure")
        self.init_error = error
        with self.assertRaises(OSError) as raised:
            self.invoke()
        self.assertIs(raised.exception, error)
        self.assertIn("tracker-finish", self.events)
        self.assertEqual(self.installs, [])

    def test_application_exit_survives_tracker_cleanup_failure(self):
        self.spec = DiagnosticSpec(**{**self.spec.__dict__, "scenario": "application_failure"})
        self.finish_error = OSError("fake finish failure")
        with self.assertRaises(SystemExit) as raised:
            self.invoke()
        self.assertEqual(raised.exception.code, 23)
        self.assertIs(self.installs[0][1]["application_error"], raised.exception)
        self.assertIsNone(self.installs[0][0])

    def test_cleanup_failure_without_application_is_propagated(self):
        error = OSError("fake finish failure")
        self.finish_error = error
        with self.assertRaises(OSError) as raised:
            self.invoke()
        self.assertIs(raised.exception, error)
        self.assertIs(self.installs[0][1]["application_error"], error)

    def test_foreign_imported_module_is_rejected_before_world_init(self):
        self.shadow_module = "levanter.tracker.wandb"
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertNotIn("world-init", self.events)
        self.assertIn("native-finish", self.events)

    def test_existing_native_run_is_not_reused_or_finished(self):
        self.existing_run = object()
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertNotIn("online-init", self.events)
        self.assertNotIn("native-finish", self.events)

    def test_existing_distributed_world_is_not_reused(self):
        self.initialized = True
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertNotIn("world-init", self.events)
        self.assertEqual(self.installs, [])

    def test_mismatched_worker_identity_prevents_world_init(self):
        self.worker_id = "different-worker"
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertNotIn("world-init", self.events)
        self.assertEqual(self.installs, [])

    def test_nonzero_rank_incomplete_tracker_becomes_application_failure(self):
        self.rank = 1
        self.report_state = "incomplete"
        with self.assertRaisesRegex(RuntimeError, "required diagnostic tracker") as raised:
            self.invoke()
        self.assertIs(self.installs[0][1]["application_error"], raised.exception)
        self.assertEqual(self.installs[0][0].state, "incomplete")

    def test_original_exit_survives_incomplete_report(self):
        self.spec = DiagnosticSpec(**{**self.spec.__dict__, "scenario": "application_failure"})
        self.report_state = "incomplete"
        with self.assertRaises(SystemExit) as raised:
            self.invoke()
        self.assertEqual(raised.exception.code, 23)
        self.assertIs(self.installs[0][1]["application_error"], raised.exception)

    def test_actual_installer_writes_separate_bound_finalization_trace(self):
        source = CANDIDATE / "lib/levanter/tests/shutdown_candidate_checks/check_hook_integration.py"
        spec = importlib.util.spec_from_file_location("entrypoint_hook_fixture", source)
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        handlers = fixture.Hooks()
        namespace, effects = fixture.load(handlers, fixture.Coordinator())
        namespace["get_job_info"] = lambda: SimpleNamespace(job_id=SimpleNamespace(to_wire=lambda: "synthetic-job"))
        previous_importer = self.fake_import

        def external_boundary(name):
            module = previous_importer(name)
            if name == "levanter.distributed":
                module.install_coordinated_iris_finalizer = namespace["install_coordinated_iris_finalizer"]
            elif name == "levanter.tracker.finalization":
                module.required_sinks_complete = fixture.receipt.required_sinks_complete
            elif name == "levanter.distributed_finalization":
                module.FinalizationObservation = fixture.core.FinalizationObservation
            elif name == "levanter.tracker.background":
                old = module.BackgroundTracker

                class Tracker(old):
                    def finish_with_report(self):
                        super().finish_with_report()
                        return fixture.good()

                module.BackgroundTracker = Tracker
            return module

        self.fake_import = external_boundary
        self.invoke()
        handlers.run()
        records = [json.loads(line) for line in (self.root / "output/finalization.jsonl").read_text().splitlines()]
        names = [row["event"] for row in records]
        self.assertLess(names.index("read_return"), names.index("shutdown_enter"))
        self.assertEqual(names[-1], "hook_return")
        self.assertEqual(records[0]["identity"]["attempt_uid"], "synthetic-attempt")
        self.assertEqual(records[0]["identity"]["execution_id"], "synthetic-job")
        self.assertEqual(effects, ["shutdown"])
        self.assertTrue(all(row["identity"] == records[0]["identity"] for row in records))


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    (RESULT_DIR / "cpu-entrypoint-checks.json").open("x").write(
        json.dumps(
            {
                "passed": result.wasSuccessful(),
                "tests": result.testsRun,
                "scope": "draft orchestration with fake import/service boundaries; no runtime package imports",
                "runtime_qualified": False,
                "new_jobs": 0,
                "numerical_kernels": 0,
            },
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)
