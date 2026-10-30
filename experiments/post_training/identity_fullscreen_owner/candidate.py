# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind each screen task to the parent's exact candidate and output namespace."""

import hashlib
import json
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for


def option(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def validate_candidate(candidate: dict) -> None:
    root = candidate["artifact_root"].rstrip("/")
    if not root.startswith("s3://marin-rno2a/tmp/ttl=14d/"):
        raise ValueError("Screen candidate requires an explicit fresh RNO14d output namespace")
    argv = candidate["screen"]["generation_argv"]
    if option(argv, "--model") != candidate["model_path"]:
        raise ValueError("Generation model differs from the parent candidate")
    expected = f"_pathb_fullscreen/outputs/{candidate['label']}/"
    if not option(argv, "--out").startswith(expected):
        raise ValueError("Generation output is outside its collected candidate namespace")
    trials = [spec["trial"] for spec in candidate["math"]]
    if sorted(trials) != ["17", "29", "greedy"]:
        raise ValueError("Screen owner requires the three bound math trials")
    for spec in candidate["math"]:
        if option(spec["argv"], "--model") != candidate["model_path"]:
            raise ValueError("Math model differs from the parent candidate")
        output = spec["output"].rstrip("/")
        if option(spec["argv"], "--output").rstrip("/") != output:
            raise ValueError("Math argv differs from its bound output")
        if output != f"{root}/math/{spec['trial']}":
            raise ValueError("Math output is outside the parent candidate namespace")


def candidate_payload(candidate: dict) -> tuple[str, bytes, str]:
    validate_candidate(candidate)
    raw = (json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    return f"_pathb_execution/candidates/{digest}.json", raw, digest


def load_candidate(path: Path, expected_sha256: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Child candidate payload differs from the parent SHA256")
    candidate = json.loads(raw)
    validate_candidate(candidate)
    return candidate


def require_fresh_namespace(candidate: dict) -> None:
    """Refuse an already populated owner prefix before allocating its model serve."""
    validate_candidate(candidate)
    fs, key = filesystem_for(candidate["artifact_root"])
    if fs.exists(key):
        raise FileExistsError("Screen owner output namespace already exists; no model job was submitted")
