# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure negative controls for the draft manifest validator; no runtime imports."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from check_manifest import load_manifest, unique_object

ROOT = Path(__file__).resolve().parent
CANDIDATE = Path(os.environ["EMA_GRAM_CANDIDATE_ROOT"]).resolve()
RESULT_DIR = Path(os.environ["EMA_GRAM_CHECK_RESULT_DIR"]).resolve()
if not RESULT_DIR.is_dir():
    raise ValueError("check result directory must already exist")
TEMPLATE = json.loads((ROOT / "cpu-stage1-manifest-template.json").read_text())


class Checks(unittest.TestCase):
    def assess(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(data))
            return load_manifest(path, CANDIDATE)[1]

    def test_template_is_valid_but_not_ready_or_authorized(self):
        result = self.assess(TEMPLATE)
        self.assertTrue(result["manifest_schema_and_source_valid"])
        self.assertFalse(result["launch_ready"])
        self.assertFalse(result["execution_authorized"])
        self.assertFalse(result["runtime_qualified"])
        self.assertEqual(result["pending_tracking_bindings"], ["entity"])
        self.assertEqual(len(result["pending_deployment_bindings"]), 3)

    def test_offline_rejected(self):
        data = copy.deepcopy(TEMPLATE)
        data["wandb"]["mode"] = "offline"
        with self.assertRaises(ValueError):
            self.assess(data)

    def test_accelerator_scope_rejected(self):
        data = copy.deepcopy(TEMPLATE)
        data["resource_scope"]["device"] = "gpu"
        with self.assertRaises(ValueError):
            self.assess(data)

    def test_boolean_health_budget_rejected(self):
        data = copy.deepcopy(TEMPLATE)
        data["communication_health"]["finalization_rendezvous_ms"] = True
        with self.assertRaises(ValueError):
            self.assess(data)

    def test_missing_source_binding_rejected(self):
        data = copy.deepcopy(TEMPLATE)
        data["source_hashes"].pop(next(iter(data["source_hashes"])))
        with self.assertRaises(ValueError):
            self.assess(data)

    def test_authorization_field_cannot_enable_execution(self):
        data = copy.deepcopy(TEMPLATE)
        data["execution_authorized"] = True
        with self.assertRaises(ValueError):
            self.assess(data)

    def test_boolean_resource_count_rejected(self):
        data = copy.deepcopy(TEMPLATE)
        data["resource_scope"]["processes_per_task"] = True
        with self.assertRaises(ValueError):
            self.assess(data)

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):
            json.loads('{"mode":"online","mode":"offline"}', object_pairs_hook=unique_object)

    def test_parent_commit_cannot_claim_current_bytes(self):
        data = copy.deepcopy(TEMPLATE)
        data["candidate_commit"] = "689fe209f8102a588ebac1bc59cf04289137a3ad"
        with self.assertRaisesRegex(ValueError, "declared candidate commit"):
            self.assess(data)

    def test_blank_entity_rejected(self):
        data = copy.deepcopy(TEMPLATE)
        data["wandb"]["entity"] = " "
        with self.assertRaises(ValueError):
            self.assess(data)

    def test_nonstring_deployment_binding_rejected(self):
        data = copy.deepcopy(TEMPLATE)
        data["deployment"]["region"] = 7
        with self.assertRaises(ValueError):
            self.assess(data)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (RESULT_DIR / "manifest-contract-checks.json").open("x").write(
        json.dumps(
            {
                "checks": result.testsRun,
                "passed": result.wasSuccessful(),
                "failures": len(result.failures),
                "errors": len(result.errors),
                "scope": "pure manifest/source checks; no runtime imports or services",
            },
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(not result.wasSuccessful())
