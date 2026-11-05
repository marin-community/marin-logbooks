# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan or execute approved disposal of finished study runs' native tensor state."""

import argparse
import hashlib
import json
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for
from verify_export import verify

SOURCE_BASE = "s3://marin-rno2a/tmp/ttl=14d/skyrl/marin/users/romain/checkpoints/sc-default-study/"
INITIAL = "s3://marin-rno2a/marin/users/dml/models/" "grug-67b-a2b-datakit-sft-262k-2026.10.05-8c76b22a/2026.10.22/hf"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def members(uri):
    fs, prefix = filesystem_for(uri)
    fs.invalidate_cache()
    return {
        "s3://" + key: (info["size"], info["ETag"], str(info["LastModified"]))
        for key, info in fs.find(prefix, detail=True).items()
        if info["type"] == "file"
    }


def verify_model(entry, run, directory):
    previous = read(directory / entry["verified_model_export_proof"])
    current = verify(
        entry["preserved_run_root"],
        INITIAL,
        run["source_revision"],
        previous.get("export_terminal_uri"),
        previous.get("export_iris_job_id", run["export_job"]),
    )
    for field in (
        "model_export",
        "model_manifest_sha256",
        "weight_shards",
        "weight_shard_bytes",
    ):
        if current[field] != previous[field]:
            raise ValueError(f"Verified final model changed: {entry['arm']}/{entry['seed']}/{field}")
    identity_fields = ("shape", "initial_float32_sha256", "final_float32_sha256")
    before = {row["name"]: [row[field] for field in identity_fields] for row in previous["sampled_weight_checks"]}
    after = {row["name"]: [row[field] for field in identity_fields] for row in current["sampled_weight_checks"]}
    if before != after:
        raise ValueError(f"Sampled final weights changed: {entry['arm']}/{entry['seed']}")
    return current


def checkpoint_prefix(entry, run):
    expected = SOURCE_BASE + f"{run['arm']}-seed{run['seed']}/{run['version']}/checkpoints/global_step_50/"
    if entry["native_checkpoint"] != expected:
        raise ValueError("Checkpoint escaped the exact owned run prefix")
    return expected


def plan(proposal_path, manifest_path, output):
    proposal, manifest = read(proposal_path), read(manifest_path)
    runs = []
    for entry in proposal["entries"]:
        (run,) = [
            row
            for row in manifest["runs"]
            if (row["arm"], row["seed"], row["version"]) == (entry["arm"], entry["seed"], entry["version"])
        ]
        if not run["eligible"]:
            raise ValueError("Only a fully admitted finished run can enter this plan")
        expected_prefix = checkpoint_prefix(entry, run)
        expected = {row["uri"]: (row["bytes"], row["etag"], row["last_modified"]) for row in entry["files"]}
        if members(expected_prefix) != expected:
            raise ValueError("Native checkpoint changed since the reviewed inventory")
        model = verify_model(entry, run, manifest_path.parent)
        tensors = [row for row in entry["files"] if row["uri"].endswith(".distcp")]
        if len(tensors) != 32 or not all(row["uri"].startswith(expected_prefix + "policy/") for row in tensors):
            raise ValueError("Expected exactly the 32 native policy tensor shards")
        runs.append(
            {
                **entry,
                "delete": tensors,
                "retain_metadata": [row for row in entry["files"] if row not in tensors],
                "fresh_model_verification": model,
            }
        )
    result = {
        "status": "read-only plan; no deletion performed",
        "proposal_sha256": hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "runs": runs,
        "delete_bytes": sum(row["bytes"] for run in runs for row in run["delete"]),
        "delete_objects": sum(len(run["delete"]) for run in runs),
        "retained": "All verified final HF weights, answers, results, telemetry and compact native metadata.",
        "irreversible_loss": "Native tensor/optimizer state cannot be resumed; this requires explicit user approval.",
    }
    write(output, result)
    return {key: value for key, value in result.items() if key != "runs"}


def discard(plan_path, approval_path, manifest_path, receipts):
    proposal, approval, manifest = read(plan_path), read(approval_path), read(manifest_path)
    if not approval["approved"] or approval["policy"] != "sc-default-study-final-models-and-results":
        raise ValueError("Explicit approval of the changed preservation policy is required")
    if approval["reviewed_proposal_sha256"] != proposal["proposal_sha256"]:
        raise ValueError("Approval differs from the reviewed checkpoint inventory")
    receipts.mkdir(exist_ok=True, parents=True)
    for entry in proposal["runs"]:
        (run,) = [
            row
            for row in manifest["runs"]
            if (row["arm"], row["seed"], row["version"]) == (entry["arm"], entry["seed"], entry["version"])
        ]
        if not run["eligible"]:
            raise ValueError("Run is no longer admitted")
        prefix = checkpoint_prefix(entry, run)
        tensors = [row for row in entry["files"] if row["uri"].endswith(".distcp")]
        if entry["delete"] != tensors or len(tensors) != 32:
            raise ValueError("Disposal targets differ from the planned native tensor shards")
        if not all(row["uri"].startswith(prefix + "policy/") for row in tensors):
            raise ValueError("A tensor shard escaped its exact native policy prefix")
        verify_model(entry, run, manifest_path.parent)
        current = members(entry["native_checkpoint"])
        expected = {row["uri"]: (row["bytes"], row["etag"], row["last_modified"]) for row in entry["files"]}
        receipt = receipts / f"{run['arm']}-{run['seed']}-{run['version']}.json"
        intent = {
            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "approval_sha256": hashlib.sha256(approval_path.read_bytes()).hexdigest(),
            "delete": entry["delete"],
            "preserved_model": entry["fresh_model_verification"],
        }
        if receipt.exists() and any(read(receipt)[key] != value for key, value in intent.items()):
            raise ValueError("Existing disposal receipt belongs to another plan or approval")
        if not receipt.exists() and current != expected:
            raise ValueError("Native checkpoint changed before disposal")
        if any(uri not in expected or signature != expected[uri] for uri, signature in current.items()):
            raise ValueError("Remaining native objects changed")
        retained = {row["uri"]: expected[row["uri"]] for row in entry["retain_metadata"]}
        if any(current.get(uri) != signature for uri, signature in retained.items()):
            raise ValueError("Required native metadata changed or disappeared")
        write(receipt, {**intent, "status": "approved disposal in progress"})
        targets = [row["uri"] for row in entry["delete"] if row["uri"] in current]
        if targets:
            fs, _ = filesystem_for(entry["native_checkpoint"])
            fs.rm([uri.removeprefix("s3://") for uri in targets])
        remaining = members(entry["native_checkpoint"])
        if remaining != retained:
            raise ValueError("Disposal did not preserve exactly the planned metadata")
        write(receipt, {**intent, "status": "complete", "retained_metadata": entry["retain_metadata"]})
        print(
            json.dumps(
                {"run": f"{run['arm']}/{run['seed']}", "disposed_bytes": sum(r["bytes"] for r in entry["delete"])}
            )
        )
    return {"completed_runs": len(proposal["runs"]), "disposed_bytes": proposal["delete_bytes"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser("plan")
    preview.add_argument("proposal", type=Path)
    preview.add_argument("manifest", type=Path)
    preview.add_argument("output", type=Path)
    removal = commands.add_parser("discard")
    removal.add_argument("plan", type=Path)
    removal.add_argument("approval", type=Path)
    removal.add_argument("manifest", type=Path)
    removal.add_argument("receipts", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        result = plan(args.proposal, args.manifest, args.output)
    else:
        result = discard(args.plan, args.approval, args.manifest, args.receipts)
    print(json.dumps(result))
