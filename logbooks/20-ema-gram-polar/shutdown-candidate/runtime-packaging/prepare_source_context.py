# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare an isolated local source checkout; never build or import runtime code."""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


def git_env():
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null", GIT_TERMINAL_PROMPT="0")
    return env


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, env=git_env()).strip()


def prepare(source, commit, destination):
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("full commit required")
    if git(source, "rev-parse", "HEAD") != commit:
        raise ValueError("source HEAD differs from requested commit")
    if git(source, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("tracked source changes must be resolved before packaging")
    if destination.exists() or destination.is_symlink():
        raise ValueError("destination must be new")
    # A file URL plus --no-local uses the Git transport, copying no shared
    # worktree metadata, object hard links, environment, or untracked files.
    subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "clone",
            "--no-local",
            "--no-checkout",
            "--depth=1",
            source.as_uri(),
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=git_env(),
    )
    if git(destination, "rev-parse", "HEAD") != commit:
        raise ValueError("source changed during clone; preserve destination for inspection")
    subprocess.run(
        ["git", "-C", str(destination), "-c", "core.hooksPath=/dev/null", "checkout", "--detach", commit],
        check=True,
        capture_output=True,
        text=True,
        env=git_env(),
    )
    if not (destination / ".git").is_dir():
        raise ValueError("checkout has no self-contained Git directory")
    if (destination / ".git/objects/info/alternates").exists():
        raise ValueError("external Git object storage is forbidden")
    if (destination / ".venv").exists() or (destination / ".venv").is_symlink():
        raise ValueError("source context must not contain a virtual environment")
    # Remove the source remote so future packaging cannot fetch or push it by accident.
    subprocess.run(["git", "-C", str(destination), "remote", "remove", "origin"], check=True, env=git_env())
    tree = git(destination, "rev-parse", "HEAD^{tree}")
    return {
        "commit": commit,
        "tree": tree,
        "source_context": str(destination),
        "self_contained_git": True,
        "shallow": git(destination, "rev-parse", "--is-shallow-repository") == "true",
        "tracked_status": git(destination, "status", "--porcelain", "--untracked-files=no"),
        "packaging_ready": False,
        "remaining": [
            "Harness inclusion",
            "Submodule and LFS materialization audit",
            "Dependency and image preparation",
        ],
        "runtime_imports": False,
        "image_built": False,
        "execution_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.audit.exists():
        raise ValueError("audit must be new")
    result = prepare(args.source, args.commit, args.destination)
    result["preparer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.audit.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
