# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap the frozen ARM profile and run the bounded host allocator probe."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as content:
        return hashlib.file_digest(content, "sha256").hexdigest()


def validate_payload(manifest_path: Path, expected: str, workspace: Path) -> dict:
    if digest(manifest_path) != expected:
        raise ValueError("Submission manifest digest changed")
    manifest = json.loads(manifest_path.read_text())
    for field in ("payload", "managed_inputs"):
        for path, expected_hash in manifest[field].items():
            if digest(workspace / path) != expected_hash:
                raise ValueError(f"Frozen {field} changed: {path}")
    if digest(workspace / manifest["optimizer_payload"]) != manifest["optimizer_sha256"]:
        raise ValueError("Frozen MuonH source digest changed")
    return manifest


def bootstrap(args, workspace: Path) -> None:
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ValueError("Probe requires Linux aarch64 and the frozen ARM profile")
    work = Path(tempfile.mkdtemp(prefix="hero-host-allocator-", dir="/cache"))
    environment = work / "environment"
    activation = work / "runtime.env"
    project = workspace / "vendor/skyrl"
    subprocess.run(
        [
            "bash",
            str(project / "cloud/iris/bootstrap_runtime.sh"),
            str(project),
            str(environment),
            str(activation),
            "megatron",
            "production",
        ],
        check=True,
    )
    command = [
        "bash",
        "-c",
        'set -euo pipefail; source "$1"; exec "$2" "$3" --execute --manifest "$4" ' '--manifest-sha256 "$5" --work "$6"',
        "hero-host-allocator",
        str(activation),
        str(environment / "bin/python"),
        str(Path(__file__).resolve()),
        str(args.manifest),
        args.manifest_sha256,
        str(work),
    ]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    os.execvpe(command[0], command, env)


def execute(args, workspace: Path, manifest: dict) -> None:
    output = Path(os.environ.get("IRIS_OUTPUT_DIR", str(args.work / "output")))
    output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "result": "preflight",
        "manifest_sha256": args.manifest_sha256,
        "source_revision": manifest["source_revision"],
        "optimizer_source_sha256": manifest["optimizer_sha256"],
        "managed_core": {name: metadata.version(name) for name in manifest["expected_core"]},
        "scope": manifest["scope"],
    }
    destination = output / "validation.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    if receipt["managed_core"] != manifest["expected_core"]:
        raise ValueError("Managed core dependencies differ from the exact frozen profile")
    command = [
        sys.executable,
        str(workspace / manifest["probe_payload"]),
        "--case",
        "all",
        "--optimizer-source",
        str(workspace / manifest["optimizer_payload"]),
        "--optimizer-sha256",
        manifest["optimizer_sha256"],
        "--output",
        str(output / "cases"),
    ]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    subprocess.run(command, env=env, check=True, timeout=1200, cwd=args.work)
    cases = {}
    for path in sorted((output / "cases").glob("*.json")):
        case = json.loads(path.read_text())
        if case["result"] != "passed" or case["completed_storage_cycles"] != 2:
            raise ValueError(f"Allocator case did not pass: {path.name}")
        cases[path.stem] = {"sha256": digest(path), "result": case["result"]}
    if len(cases) != 6:
        raise ValueError("Allocator qualification requires six fresh-process cases")
    receipt.update(result="passed", cases=cases)
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"result": "passed", "output": str(output)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--check-staged", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    workspace = args.manifest.parent
    manifest = validate_payload(args.manifest, args.manifest_sha256, workspace)
    if args.check_staged:
        print(
            json.dumps(
                {
                    "result": "staged-payload-valid",
                    "workspace": str(workspace),
                    "source_revision": manifest["source_revision"],
                }
            )
        )
        return
    if args.execute:
        if args.work is None:
            raise ValueError("Execution requires the owned bootstrap work directory")
        execute(args, workspace, manifest)
        return
    bootstrap(args, workspace)


if __name__ == "__main__":
    main()
