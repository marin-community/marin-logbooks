# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source-root bootstrap controls; runtime runner is always a fake."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from worker_main import execute_prepared, pin_source_roots


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for package in ("levanter", "iris"):
            file = self.root / "lib" / package / "src" / package / "__init__.py"
            file.parent.mkdir(parents=True)
            file.write_text('raise AssertionError("package must not be imported during bootstrap")\n')
        self.original_path = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), self.original_path))

    def test_pins_without_importing_packages(self):
        roots = pin_source_roots(self.root)
        self.assertEqual(sys.path[:2], roots)
        self.assertNotIn("iris", sys.modules)
        self.assertNotIn("levanter", sys.modules)

    def test_existing_runtime_module_rejected(self):
        with patch.dict(sys.modules, {"jax": SimpleNamespace()}):
            with self.assertRaises(RuntimeError):
                pin_source_roots(self.root)
        self.assertEqual(sys.path, self.original_path)

    def test_unresolved_manifest_never_calls_runner(self):
        calls = []
        with self.assertRaises(ValueError):
            execute_prepared(
                SimpleNamespace(candidate_root=self.root),
                {"pending_tracking_bindings": ["entity"], "pending_deployment_bindings": []},
                self.root / "out",
                runner=lambda *args: calls.append(args),
            )
        self.assertEqual(calls, [])
        self.assertEqual(sys.path, self.original_path)

    def test_resolved_bindings_reach_only_fake_runner(self):
        calls = []
        spec = SimpleNamespace(candidate_root=self.root)
        execute_prepared(
            spec,
            {"pending_tracking_bindings": [], "pending_deployment_bindings": []},
            self.root / "out",
            runner=lambda *args: calls.append(args),
        )
        self.assertEqual(calls, [(spec, self.root / "out")])


if __name__ == "__main__":
    out = Path(os.environ["EMA_GRAM_CHECK_RESULT_DIR"])
    if not out.is_dir():
        raise ValueError("result directory must exist")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with (out / "worker-bootstrap-checks.json").open("x") as stream:
        json.dump(
            {
                "tests": result.testsRun,
                "passed": result.wasSuccessful(),
                "runtime_packages_imported": False,
                "jobs": 0,
                "kernels": 0,
            },
            stream,
            indent=2,
        )
    raise SystemExit(not result.wasSuccessful())
