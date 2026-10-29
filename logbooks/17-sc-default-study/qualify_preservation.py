# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check full-byte copying, repair and resume on existing tiny control objects."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import preserve
from rigging.filesystem.s3_compat import configure_coreweave_s3


def qualify(plan_path, execution_source_revision, output):
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != execution_source_revision:
        raise ValueError("Qualification must execute the declared source revision")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Qualification requires the published clean tree")
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    tool_sha = hashlib.sha256((directory / "preserve.py").read_bytes()).hexdigest()
    if tool_sha != plan["tool_source_sha256"]:
        raise ValueError("Preservation tool differs from the published plan")
    payloads = {
        "pattern.bin": bytes(range(251)) * 1024,
        "nested/metadata.json": b'{"kind":"sc-default-preservation-control","version":1}\n',
        "empty": b"",
    }
    source_fs, source_key = preserve.filesystem_for(plan["source_prefix"])
    before = preserve.source_members(source_fs, source_key)
    if len(before) != len(payloads):
        raise ValueError("Control source inventory differs")
    for row in plan["objects"]:
        name = row["relative_path"]
        payload = preserve.read(plan["source_prefix"] + name)
        if payload != payloads[name] or len(payload) != row["bytes"]:
            raise ValueError(f"Original control payload differs: {name}")
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError(f"Original control checksum differs: {name}")
    archive_fs, archive_key = preserve.filesystem_for(plan["archive_prefix"])
    try:
        existing = archive_fs.find(archive_key)
    except FileNotFoundError:
        existing = []
    if existing:
        raise ValueError("Control archive is not fresh; inspect before retrying")
    listing = preserve.inventory(plan["source_prefix"], plan["archive_prefix"], plan["run_source_revision"])
    manifest = json.loads(preserve.read(listing["manifest_uri"]))
    corrupt = next(row for row in manifest["files"] if row["source"].endswith("/pattern.bin"))
    preserve.immutable(corrupt["archive"], b"X" * corrupt["bytes"])
    first = preserve.preserve(listing["manifest_uri"], execution_source_revision)
    complete = json.loads(preserve.read(first["manifest_uri"]))
    repaired = [row["source"] for row in complete["files"] if row["streamed_recovery"]]
    if repaired != [corrupt["source"]]:
        raise ValueError("The wrong same-size archive member was not repaired")
    for row in complete["files"]:
        name = row["source"].removeprefix(plan["source_prefix"])
        original, archived = preserve.read(row["source"]), preserve.read(row["archive"])
        if original != archived or original != payloads[name]:
            raise ValueError(f"Source/archive round trip differs: {name}")
        if hashlib.sha256(original).hexdigest() != row["sha256"]:
            raise ValueError(f"Complete manifest checksum differs: {name}")
    second = preserve.preserve(listing["manifest_uri"], execution_source_revision)
    if first != second or preserve.source_members(source_fs, source_key) != before:
        raise ValueError("Resume changed the complete proof or source inventory")
    proof = {
        "passed": True,
        "execution_source_revision": execution_source_revision,
        "run_source_revision": plan["run_source_revision"],
        "operator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256(raw).hexdigest(),
        "tool_source_sha256": tool_sha,
        "source_prefix": plan["source_prefix"],
        "archive_prefix": plan["archive_prefix"],
        "input_manifest_uri": listing["manifest_uri"],
        "complete_manifest_uri": first["manifest_uri"],
        "complete_manifest_sha256": first["manifest_sha256"],
        "objects": len(payloads),
        "bytes": sum(map(len, payloads.values())),
        "all_source_archive_sha256_equal": True,
        "original_payloads_and_inventory_unchanged": True,
        "wrong_same_size_derived_copy_repaired": True,
        "identical_manifest_on_resume": True,
        "source_deletions": 0,
        "gpu_count": 0,
        "model_or_grading_calls": 0,
        "limits": (
            "Tiny objects verify copying, repair and resume; full native multipart preservation remains unqualified."
        ),
    }
    output.write_text(json.dumps(proof, indent=2) + "\n")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--execution-source-revision", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    print(json.dumps(qualify(args.plan, args.execution_source_revision, args.output)))
