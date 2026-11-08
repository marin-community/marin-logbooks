# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepared worker bootstrap. Execution requires separate job authorization.

Default invocation performs local preflight only. No submission API is present.
The future runtime image must contain a self-contained candidate Git checkout,
Git, and pinned CPU dependencies; a worktree .git pointer is not sufficient.
"""

import argparse
import importlib.machinery
import json
import sys
from pathlib import Path

from check_manifest import load_manifest
from cpu_entrypoint import run_diagnostic


def pin_source_roots(candidate_root: Path):
    """Pin top-level source selection without importing runtime packages."""
    for name in sys.modules:
        if any(name == prefix or name.startswith(prefix + ".") for prefix in ("jax", "iris", "levanter")):
            raise RuntimeError("worker requires a fresh process before source pinning")
    roots = []
    for package in ("levanter", "iris"):
        root = (candidate_root / "lib" / package / "src").resolve()
        expected = root / package / "__init__.py"
        spec = importlib.machinery.PathFinder.find_spec(package, [str(root)])
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
            raise RuntimeError("candidate top-level package origin missing or unexpected")
        roots.append(str(root))
    sys.path[:0] = roots
    return roots


def execute_prepared(spec, assessment, directory, *, runner=run_diagnostic):
    """Runtime seam; tests replace runner and never initialize external services."""
    if assessment["pending_tracking_bindings"] or assessment["pending_deployment_bindings"]:
        raise ValueError("worker deployment bindings remain unresolved")
    pin_source_roots(spec.candidate_root)
    runner(spec, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--execute", action="store_true", help="Requires separate explicit qualification-job authorization"
    )
    args = parser.parse_args()
    spec, assessment = load_manifest(args.manifest, args.candidate_root)
    if not args.execute:
        print(json.dumps(assessment, indent=2))
        return
    if args.output_directory is None:
        parser.error("--execute requires a fresh --output-directory for this attempt")
    execute_prepared(spec, assessment, args.output_directory)


if __name__ == "__main__":
    main()
