# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Portable CPU research diagnostic; publication and execution gates live in its records."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("baseline", "raw", "microbatch"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--devices", type=int, choices=(1, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    args.research_root = args.research_root.resolve()
    args.output = args.output.resolve()
    args.config = args.config.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "mode": args.mode,
        "source_root": str(args.source_root),
        "research_root": str(args.research_root),
        "config_path": str(args.config),
        "probe_path": str(Path(__file__).resolve()),
        "python": sys.executable,
        "requested_devices": args.devices,
        "project_imports_and_execution": "same process",
        "preflight_complete": False,
        "stage": "preflight.inputs",
        "completed_controls": {},
    }

    def checkpoint(stage, **details):
        receipt["stage"] = stage
        receipt.update(details)
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    try:
        checkpoint("preflight.inputs")
        assert Path(__file__).resolve().is_relative_to(args.research_root)
        assert args.config.is_relative_to(args.research_root)
        receipt["probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        receipt["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
        config = json.loads(args.config.read_text())
        receipt["source_commit"] = config["source_commit"]
        checkpoint("preflight.source_commit")
        observed_commit = subprocess.check_output(
            ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        receipt["observed_source_commit"] = observed_commit
        checkpoint("preflight.source_commit")
        assert observed_commit == config["source_commit"]
        checkpoint("preflight.source_cleanliness")
        status = subprocess.check_output(["git", "-C", str(args.source_root), "status", "--porcelain"], text=True)
        receipt["source_status"] = status
        checkpoint("preflight.source_cleanliness")
        assert not status.strip()
        receipt["verified_source_files"] = []
        for name, source in config["source_files"].items():
            checkpoint("preflight.source_hashes", current_source_file=name)
            assert hashlib.sha256((args.source_root / name).read_bytes()).hexdigest() == source["sha256"], name
            receipt["verified_source_files"].append(name)
        checkpoint("preflight.worker_integrity")
        worker_path = Path(__file__).resolve().parent / config["worker"]["filename"]
        assert worker_path.parent == Path(__file__).resolve().parent
        receipt["worker_path"] = str(worker_path)
        receipt["worker_sha256"] = hashlib.sha256(worker_path.read_bytes()).hexdigest()
        checkpoint("preflight.worker_integrity")
        assert receipt["worker_sha256"] == config["worker"]["sha256"]
        checkpoint("preflight.project_imports")
        specification = importlib.util.spec_from_file_location("native_jax_core_fp8_worker", worker_path)
        assert specification is not None and specification.loader is not None
        worker = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(worker)
        receipt["result"] = worker.execute(args, receipt, checkpoint)
    except Exception as error:
        receipt["failure_classification"] = (
            "invalid_harness_or_provenance"
            if not receipt["preflight_complete"]
            else "diagnostic_control_failure_review_required"
        )
        receipt["exception"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(limit=10),
        }
        checkpoint(receipt["stage"])
        raise
    receipt["execution_outcome"] = "completed"
    checkpoint("completed")


if __name__ == "__main__":
    main()
