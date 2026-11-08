# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Split a cached uv export without resolving, downloading or installing packages."""

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

TARGET = {
    "implementation_name": "cpython",
    "implementation_version": "3.12.12",
    "os_name": "posix",
    "platform_machine": "x86_64",
    "platform_python_implementation": "CPython",
    "platform_system": "Linux",
    "python_full_version": "3.12.12",
    "python_version": "3.12",
    "sys_platform": "linux",
    "extra": "",
}
WORKSPACE = frozenset(
    {
        "./lib/finelog",
        "./lib/finestore",
        "./lib/fray",
        "./lib/haliax",
        "./lib/iris",
        "./lib/levanter",
        "./lib/rigging",
        "./lib/zephyr",
    }
)


def split(text, *, expected_workspace=WORKSPACE):
    blocks = []
    pending = ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pending += line.removesuffix("\\").rstrip() + " "
        if not line.endswith("\\"):
            blocks.append(pending.strip())
            pending = ""
    if pending:
        raise ValueError("unfinished requirement continuation")
    local, selected, rejected, names = [], [], [], set()
    for block in blocks:
        if block.startswith("-e "):
            path = block[3:]
            if path not in expected_workspace or ".." in PurePosixPath(path).parts or path in local:
                raise ValueError("unexpected or duplicate workspace path")
            local.append(path)
            continue
        parts = block.split(" --hash=")
        req = Requirement(parts[0])
        specs = list(req.specifier)
        if req.url or len(specs) != 1 or specs[0].operator != "==" or "*" in specs[0].version:
            raise ValueError("exact registry version required")
        if len(parts) < 2 or any(re.fullmatch(r"sha256:[0-9a-f]{64}", h.strip()) is None for h in parts[1:]):
            raise ValueError("explicit SHA256 artifact hashes required")
        if req.marker:
            # Do not borrow missing host facts from packaging's default environment.
            if any(key in str(req.marker) for key in ("platform_release", "platform_version")):
                raise ValueError("unbound target marker")
            if not req.marker.evaluate(TARGET):
                rejected.append(str(req))
                continue
        canonical = canonicalize_name(req.name)
        if canonical in names:
            raise ValueError("duplicate selected distribution")
        names.add(canonical)
        selected.append(block)
    if set(local) != set(expected_workspace):
        raise ValueError("workspace coverage differs from bound source inventory")
    return selected, local, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--export-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.export.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.export_sha256:
        raise ValueError("cached export digest mismatch")
    registry, workspace, excluded = split(raw.decode())
    args.output.mkdir(exist_ok=False)
    (args.output / "registry-requirements.txt").write_text("\n".join(registry) + "\n")
    (args.output / "workspace-editables.txt").write_text("\n".join("-e " + p for p in workspace) + "\n")
    files = ["registry-requirements.txt", "workspace-editables.txt"]
    manifest = {
        "target": TARGET,
        "registry_count": len(registry),
        "workspace_count": len(workspace),
        "excluded": excluded,
        "input_sha256": args.export_sha256,
        "files_sha256": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest() for name in files},
        "installed": False,
        "wheel_compatibility_verified": False,
        "runtime_qualified": False,
    }
    (args.output / "split-audit.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
