# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local/worker Stage-A controller for the existing full Trainer diagnostic."""

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "2b17db78aedc0a603664238fbfa0bc0c68542d2b"
LIMITS = {"files": 128, "per_file_bytes": 32 * 1024**2, "raw_bytes": 128 * 1024**2, "gzip_bytes": 16 * 1024**2}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retention(directory):

    if any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("Symlink in retained artifacts")
    paths = sorted(p for p in directory.rglob("*") if p.is_file())
    if len(paths) > LIMITS["files"]:
        raise ValueError("Artifact count cap exceeded")
    total = 0
    sink = io.BytesIO()
    with gzip.GzipFile(fileobj=sink, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for path in paths:
                name = path.name
                if not (
                    name.endswith(".json") or name.endswith(".stablehlo.txt") or name.endswith(".optimized_hlo.txt")
                ):
                    raise ValueError(f"Disallowed retained artifact: {path}")
                size = path.stat().st_size
                total += size
                if size > LIMITS["per_file_bytes"] or total > LIMITS["raw_bytes"]:
                    raise ValueError("Complete artifact exceeds retention cap; no truncation")
                archive.add(path, arcname=str(path.relative_to(directory)), recursive=False)
    if len(sink.getvalue()) > LIMITS["gzip_bytes"]:
        raise ValueError("Compressed artifact cap exceeded; no truncation")
    return {
        "files": len(paths),
        "raw_bytes": total,
        "gzip_bytes": len(sink.getvalue()),
        "limits": LIMITS,
        "archive_written": False,
        "storage_operation": False,
    }


def compare(anchor, instrumented):
    for kind in ("legacy", "native"):
        a = anchor["variants"][kind]
        b = instrumented["variants"][kind]
        for label in ("loss", "gradients", "raw_gradient_leaves"):
            if a[label] != b[label]:
                raise ValueError(f"Fatal instrumentation identity failure: {kind}/{label}")
        if a["layouts"] != b["layouts"]:
            raise ValueError(f"Fatal instrumentation physical placement failure: {kind}")
    operands = instrumented["operands"]
    equality = {}
    for label in ("X", "G"):
        a, b = (operands[k][label] for k in ("legacy", "native"))
        for field in ("spec", "mesh", "memory_kind"):
            if a[field] != b[field]:
                raise ValueError(f"Cross-branch operand physical placement differs: {label}/{field}")
        if [{k: v for k, v in s.items() if k not in ("callbacks", "value")} for s in a["shards"]] != [
            {k: v for k, v in s.items() if k not in ("callbacks", "value")} for s in b["shards"]
        ]:
            raise ValueError(f"Cross-branch operand device/shard assignment differs: {label}")
        equality[label] = a["global"] == b["global"] and [s["value"] for s in a["shards"]] == [
            s["value"] for s in b["shards"]
        ]
    return {
        "instrumentation_loss_all_raw_gradients_exact": True,
        "canonical_operands_exact": equality,
        "branch": (
            "upstream_operand_mismatch"
            if not all(equality.values())
            else "operands_identical_local_vjp_or_reduction_unresolved"
        ),
        "local_contraction_cause_established": False,
        "stage_b_implemented": False,
        "normal_executable_or_trajectory_conclusion": False,
        "model_adoption": False,
    }


def controller(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "native-jax-oproj-stage-a.v1",
        "completed": False,
        "scientific_acceptance": False,
        "mode": args.mode,
        "source_commit": SOURCE,
        "children": [],
    }
    failure = None
    try:
        for phase in ("anchor", "instrumented"):
            child_output = output / phase
            child_output.mkdir()
            command = [
                sys.executable,
                str(Path(__file__).with_name("child.py").resolve()),
                "--mode",
                args.mode,
                "--phase",
                phase,
                "--output",
                str(child_output),
                "--config",
                str(args.config.resolve()),
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                [
                    str(ROOT / "lib/levanter/src"),
                    str(ROOT / "lib/haliax/src"),
                    str(ROOT / "lib/fray/src"),
                    str(ROOT / "lib/marin/src"),
                    str(ROOT / "lib/iris/src"),
                ]
            )
            # pycache is outside retained evidence.
            env["PYTHONPYCACHEPREFIX"] = tempfile.mkdtemp(prefix=f"oproj-{phase}-pycache-")
            result = subprocess.run(command, env=env, check=False, timeout=1800)
            report["children"].append({"phase": phase, "command": command, "exit_code": result.returncode})
            if result.returncode:
                raise ValueError(f"{phase} child failed with exit {result.returncode}")
        anchor = json.loads((output / "anchor/report.json").read_text())
        instrumented = json.loads((output / "instrumented/report.json").read_text())
        report["identity"] = compare(anchor, instrumented)
        report["completed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        failure = error
    finally:
        write_json(output / "report.json", report)
        try:
            receipt = retention(output)
            write_json(output / "retention.json", receipt)
        except Exception as error:
            report["completed"] = False
            report["retention_error"] = f"{type(error).__name__}: {error}"
            write_json(output / "report.json", report)
            if failure is None:
                failure = error
    if failure is not None:
        raise failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run",))
    parser.add_argument("--mode", required=True, choices=("cpu", "target"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    controller(args)


if __name__ == "__main__":
    main()
