# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One native serving task captures the fixed200 A5b pilot, never a full pool."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.post_training.identity_path_b.a5 import execute as parent
from experiments.post_training.identity_path_b.a5b.data import QUOTAS, TEACHER_SYSTEM
from experiments.post_training.identity_path_b.replay_capture import T2_CAPTURE_SHA256
from experiments.post_training.identity_t2 import capture as t2

BINDING = Path(__file__).with_name("pilot-binding.json")


def binding() -> dict:
    return json.loads(BINDING.read_text())


def checked_manifest(path: Path, expected_sha256: str, shard: int) -> list[dict]:
    bound = binding()
    if hashlib.sha256(Path(t2.__file__).read_bytes()).hexdigest() != T2_CAPTURE_SHA256:
        raise ValueError("Known-good T2 capture primitive changed")
    raw = path.read_bytes()
    if (
        shard != 0
        or expected_sha256 != bound["manifest"]["sha256"]
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise ValueError("Require the exact authorized200-source A5b pilot manifest")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 200 or [row["source_id"] for row in rows] != bound["source_ids"]:
        raise ValueError("Pilot originals or order changed")
    if Counter(row["family"] for row in rows) != QUOTAS:
        raise ValueError("Pilot family counts changed")
    for row in rows:
        if (
            row["teacher_system"] != TEACHER_SYSTEM
            or row["teacher"] != "held_profile_user_only"
            or row["tools"]
            or row["enable_thinking"] is not True
            or not row["prompt"]
            or any(message["role"] != "user" or set(message) != {"role", "content"} for message in row["prompt"])
            or row["sampling_seed"]
            != int(hashlib.sha256(f"310:A5:{row['source_id']}".encode()).hexdigest()[:8], 16) % (2**31)
        ):
            raise ValueError("Pilot teacher/settings/ordered USER-only context changed")
    return rows


def validate_args(run_id: str, artifact_root: str, manifest_uri: str, manifest_sha256: str, shard: int) -> None:
    bound = binding()
    if (run_id, artifact_root, manifest_uri, manifest_sha256, shard) != (
        bound["run_id"],
        bound["artifact_root"],
        bound["manifest"]["uri"],
        bound["manifest"]["sha256"],
        0,
    ):
        raise ValueError("A5b execution differs from the fixed pilot binding")


def submission(
    run_id: str, artifact_root: str, manifest_uri: str, manifest_sha256: str, shard: int, credentials: dict
) -> dict:
    validate_args(run_id, artifact_root, manifest_uri, manifest_sha256, shard)
    request = parent.same_task_submission(run_id, artifact_root, manifest_uri, manifest_sha256, shard, credentials)
    old = "experiments.post_training.identity_path_b.a5.execute "
    command = request["entrypoint"].command[-1]
    if command.count(old) != 1:
        raise ValueError("Unexpected native capture entrypoint")
    request["entrypoint"].command[-1] = command.replace(old, "experiments.post_training.identity_path_b.a5b.execute ")
    return request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--shard", type=int, choices=(0,), required=True)
    args = parser.parse_args()
    validate_args(args.run_id, args.artifact_root, args.manifest_uri, args.manifest_sha256, args.shard)
    parent.run_same_task(
        args.run_id,
        args.artifact_root,
        args.manifest_uri,
        args.manifest_sha256,
        args.shard,
        manifest_loader=checked_manifest,
    )


if __name__ == "__main__":
    main()
