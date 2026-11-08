# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure fixture checks for pinned registry/source separation."""

import unittest

from split_export import split

HASH = " --hash=sha256:" + "a" * 64


class Checks(unittest.TestCase):
    def test_target_selection_preserves_hash_and_separates_workspace(self):
        registry, local, excluded = split(
            '-e ./lib/iris\nexample==1; sys_platform == "linux"'
            + HASH
            + '\nexample==2; sys_platform == "darwin"'
            + HASH,
            expected_workspace={"./lib/iris"},
        )
        self.assertEqual(len(registry), 1)
        self.assertTrue(registry[0].endswith(HASH))
        self.assertEqual(local, ["./lib/iris"])
        self.assertEqual(len(excluded), 1)

    def test_unpinned_version_rejected(self):
        with self.assertRaises(ValueError):
            split("example>=1" + HASH, expected_workspace=set())

    def test_missing_hash_rejected(self):
        with self.assertRaises(ValueError):
            split("example==1", expected_workspace=set())

    def test_unknown_workspace_rejected(self):
        with self.assertRaises(ValueError):
            split("-e ../other", expected_workspace={"./lib/iris"})

    def test_host_dependent_marker_rejected(self):
        with self.assertRaises(ValueError):
            split('example==1; platform_release == "x"' + HASH, expected_workspace=set())

    def test_equivalent_distribution_spellings_rejected(self):
        for spelling in ("example.name", "EXAMPLE__NAME", "example---name"):
            with self.subTest(spelling=spelling), self.assertRaises(ValueError):
                split(spelling + "==1" + HASH + "\nexample-name==2" + HASH, expected_workspace=set())

    def test_duplicate_selected_distribution_rejected(self):
        with self.assertRaises(ValueError):
            split("example==1" + HASH + "\nexample==2" + HASH, expected_workspace=set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
