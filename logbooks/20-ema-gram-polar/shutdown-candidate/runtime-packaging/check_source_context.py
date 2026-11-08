# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local toy Git fixtures only: no candidate runtime or package execution."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from prepare_source_context import git, git_env, prepare


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.run_git("init")
        (self.source / "source.txt").write_text("bound source\n")
        self.run_git("add", "source.txt")
        self.run_git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "fixture")
        self.commit = git(self.source, "rev-parse", "HEAD")

    def run_git(self, *args):
        subprocess.run(["git", "-C", str(self.source), *args], check=True, capture_output=True, env=git_env())

    def test_self_contained_checkout_excludes_untracked_environment(self):
        (self.source / ".venv").symlink_to(self.root / "shared-environment")
        dest = self.root / "prepared"
        result = prepare(self.source, self.commit, dest)
        self.assertTrue((dest / ".git").is_dir())
        self.assertFalse((dest / ".venv").is_symlink())
        self.assertEqual(git(dest, "show", self.commit + ":source.txt"), "bound source")
        self.assertEqual(git(dest, "remote"), "")
        self.assertFalse(result["packaging_ready"])
        self.assertFalse(result["execution_authorized"])

    def test_dirty_tracked_source_rejected_without_destination(self):
        (self.source / "source.txt").write_text("dirty\n")
        dest = self.root / "prepared"
        with self.assertRaises(ValueError):
            prepare(self.source, self.commit, dest)
        self.assertFalse(dest.exists())

    def test_wrong_commit_rejected_without_destination(self):
        dest = self.root / "prepared"
        with self.assertRaises(ValueError):
            prepare(self.source, "0" * 40, dest)
        self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
