# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only first-stage manifest CLI. It cannot initialize or submit a job."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from cpu_entrypoint import BOUND_MODULES, DiagnosticSpec, validate_source_bindings

RESOURCE_SCOPE = {
    "device": "cpu",
    "tasks": 4,
    "processes_per_task": 1,
    "cpu_per_task": 4,
    "memory_per_task": "8g",
    "disk_per_task": "8g",
    "training_steps": 0,
    "sentinels_per_rank": 4,
}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest key")
        result[key] = value
    return result


def object_fields(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError("manifest object fields differ from schema")


def nonempty_string(value):
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError("nonempty, unpadded string required")


def hex_digest(value, length):
    if type(value) is not str or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError("full lowercase hexadecimal digest required")


def bind_commit(spec):
    """Join eight declared working files to local Git commit bytes, without imports."""

    def git(*args):
        result = subprocess.run(["git", "-C", str(spec.candidate_root), *args], capture_output=True, check=False)
        if result.returncode:
            raise ValueError("candidate commit or source blob unavailable")
        return result.stdout

    if git("cat-file", "-t", spec.candidate_commit).strip() != b"commit":
        raise ValueError("candidate identity must name a commit")
    for relative, expected in spec.source_hashes.items():
        blob = git("show", f"{spec.candidate_commit}:{relative}")
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError("source digest does not belong to declared candidate commit")


def load_manifest(path: Path, candidate_root: Path):
    data = json.loads(path.read_text(), object_pairs_hook=unique_object)
    object_fields(
        data,
        {
            "schema_version",
            "campaign",
            "scenario",
            "candidate_commit",
            "source_hashes",
            "wandb",
            "communication_health",
            "resource_scope",
            "deployment",
        },
    )
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported schema version")
    nonempty_string(data["campaign"])
    if data["scenario"] not in ("clean", "application_failure"):
        raise ValueError("unsupported first-stage scenario")
    object_fields(data["resource_scope"], RESOURCE_SCOPE)
    for key, expected in RESOURCE_SCOPE.items():
        actual = data["resource_scope"][key]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError("manifest exceeds the reviewed CPU first-stage scope")
    object_fields(data["wandb"], {"entity", "project", "mode", "resume", "reinit"})
    for value in data["wandb"].values():
        nonempty_string(value)
    for key, expected in {"mode": "online", "resume": "never", "reinit": "create_new"}.items():
        if data["wandb"][key] != expected:
            raise ValueError("tracking mode or fresh-run policy differs from diagnostic contract")
    health = data["communication_health"]
    object_fields(health, {"finalization_rendezvous_ms", "justification"})
    if type(health["finalization_rendezvous_ms"]) is not int or health["finalization_rendezvous_ms"] <= 0:
        raise ValueError("explicit positive communication-health budget required")
    nonempty_string(health["justification"])
    object_fields(data["deployment"], {"region", "controller_revision", "runtime_lock_sha256"})
    for key, value in data["deployment"].items():
        if value is not None:
            nonempty_string(value)
            if key == "controller_revision":
                hex_digest(value, 40)
            elif key == "runtime_lock_sha256":
                hex_digest(value, 64)
    hex_digest(data["candidate_commit"], 40)
    object_fields(data["source_hashes"], BOUND_MODULES.values())
    for digest in data["source_hashes"].values():
        hex_digest(digest, 64)
    spec = DiagnosticSpec(
        data["campaign"],
        data["scenario"],
        data["candidate_commit"],
        candidate_root,
        data["source_hashes"],
        data["wandb"]["project"],
        data["wandb"]["entity"],
        health["finalization_rendezvous_ms"],
    )
    validate_source_bindings(spec)
    bind_commit(spec)
    pending = [key for key, value in data["deployment"].items() if value is None]
    pending_tracking = [key for key in ("entity", "project") if data["wandb"][key] == "UNRESOLVED"]
    return spec, {
        "manifest_schema_and_source_valid": True,
        "execution_authorized": False,
        "launch_ready": False,
        "runtime_qualified": False,
        "pending_tracking_bindings": pending_tracking,
        "scope": data["resource_scope"],
        "pending_deployment_bindings": pending,
        "source_binding_scope": (
            "eight declared files match local working bytes and declared Git commit; not whole environment attestation"
        ),
        "remaining_external_evidence": [
            "actual CPU worker placement and host separation",
            "installed dependency versions",
            "fresh coordinator state across attempts",
            "independent ONLINE history and natural controller outcomes",
        ],
        "wallclock_limits": "none specified; no scheduler, task or wrapper deadline",
        "preemption_retry_policy": "omit overrides and preserve Iris defaults",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    args = parser.parse_args()
    _, assessment = load_manifest(args.manifest, args.candidate_root)
    print(json.dumps(assessment, indent=2))
