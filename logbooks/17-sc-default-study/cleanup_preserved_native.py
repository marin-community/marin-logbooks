# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Remove redundant native tensors only after full-state archive verification."""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import preserve
from discard_native_state import checkpoint_prefix, members, read, verify_model, write
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

HUB = "lib/iris/config/marin.yaml"
TERMINAL_JOBS = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_KILLED"}
TERMINAL_TASKS = {"TASK_STATE_SUCCEEDED", "TASK_STATE_FAILED", "TASK_STATE_KILLED"}


def content_manifest(uri):
    raw = preserve.read(uri)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PurePosixPath(uri).stem:
        raise ValueError("Archive manifest differs from its immutable digest")
    complete = json.loads(raw)
    if complete["schema"] != "sc-default-run-preservation/3":
        raise ValueError("Expected the qualified full-content archive format")
    if not complete["all_source_archive_sha256_equal"] or not complete["source_tree_unchanged"]:
        raise ValueError("Full source/archive verification is required")
    if complete["source_deletions"] != 0:
        raise ValueError("Expected an archive created from the unchanged original source")
    return complete, digest


def stopped_job(job_id, config):
    command = ["uv", "run", "--offline", "--frozen", "iris", "--config", config, "rpc", "controller"]

    def query(operation):
        result = subprocess.run(
            [*command, operation, "--job-id", job_id], check=True, capture_output=True, text=True, timeout=60
        )
        return json.loads(result.stdout[result.stdout.index("{") :])

    status = query("get-job-status")
    request = status.pop("request", None)
    tasks = query("list-tasks")["tasks"]
    job = status["job"]
    if job["job_id"] != job_id or job["state"] not in TERMINAL_JOBS:
        raise ValueError(f"Consumer is not terminal: {job_id}")
    if not tasks or any(task["state"] not in TERMINAL_TASKS for task in tasks):
        raise ValueError(f"Consumer still has unfinished tasks: {job_id}")
    return {
        "job_id": job_id,
        "controller_config": config,
        "state": job["state"],
        "task_states": [task["state"] for task in tasks],
        "request_sha256": (
            hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest() if request is not None else None
        ),
        "observed_utc": datetime.now(UTC).isoformat(),
    }


def evaluate(complete_uri, entry, manifest_path, submission, allow_removed=False):
    complete, complete_sha = content_manifest(complete_uri)
    preserve.check_prefixes(complete["source_prefix"], complete["archive_prefix"])
    manifest = read(manifest_path)
    (run,) = [
        row
        for row in manifest["runs"]
        if (row["arm"], row["seed"], row["version"]) == (entry["arm"], entry["seed"], entry["version"])
    ]
    if not run["eligible"] or complete["run_source_revision"] != run["source_revision"]:
        raise ValueError("A matching fully admitted finished run is required")
    native = checkpoint_prefix(entry, run)
    if complete["source_prefix"] != native.removesuffix("global_step_50/"):
        raise ValueError("Archive must cover this exact run's entire checkpoint tree")
    if entry["preserved_run_root"] != run["export"].removesuffix("/exports"):
        raise ValueError("Portable model root differs from the admitted run")
    if entry["verified_model_export_proof"] != run["export_verification"]:
        raise ValueError("Portable model proof differs from the admitted run")
    if complete["tool_source_sha256"] != hashlib.sha256(Path(preserve.__file__).read_bytes()).hexdigest():
        raise ValueError("Archive code differs from the qualified preservation tool")
    if submission["source_revision"] != complete["execution_source_revision"]:
        raise ValueError("Archive execution source differs from the recorded job")
    if submission["input_manifest_uri"] != complete["input_manifest_uri"]:
        raise ValueError("Archive job used another input manifest")
    input_raw = preserve.read(complete["input_manifest_uri"])
    input_sha = hashlib.sha256(input_raw).hexdigest()
    if input_sha != complete["input_manifest_sha256"] or input_sha != submission["input_manifest_sha256"]:
        raise ValueError("Original input manifest digest changed")
    if input_sha != PurePosixPath(complete["input_manifest_uri"]).stem:
        raise ValueError("Input manifest no longer has its immutable content name")
    input_manifest = json.loads(input_raw)
    if any(
        input_manifest[field] != complete[field] for field in ("source_prefix", "archive_prefix", "run_source_revision")
    ):
        raise ValueError("Complete archive changed the original input scope")
    inputs = {row["source"]: row for row in input_manifest["files"]}
    archived = {row["source"]: row for row in complete["files"]}
    if not archived or len(archived) != len(complete["files"]) or set(inputs) != set(archived):
        raise ValueError("Complete archive omits or repeats input members")
    for uri, row in archived.items():
        if any(row[field] != value for field, value in inputs[uri].items()):
            raise ValueError("Complete archive changed an original member identity")
        if not uri.startswith(complete["source_prefix"]):
            raise ValueError("Archived source member escaped the checkpoint tree")
        if not row["archive"].startswith(complete["archive_prefix"] + "/objects/") or len(row["sha256"]) != 64:
            raise ValueError("Invalid archived member or content digest")
    original = {row["uri"]: (row["bytes"], row["etag"], row["last_modified"]) for row in entry["files"]}
    covered = {
        uri: (row["bytes"], row["etag"], row["last_modified"]) for uri, row in archived.items() if uri.startswith(native)
    }
    if covered != original:
        raise ValueError("Archive does not preserve every reviewed native checkpoint member")
    expected = {uri: (row["bytes"], row["etag"], row["last_modified"]) for uri, row in archived.items()}
    archive_expected = {row["archive"]: (row["bytes"], row["archive_etag"]) for row in archived.values()}
    archive_current = {
        uri: signature[:2] for uri, signature in members(complete["archive_prefix"] + "/objects/").items()
    }
    if archive_current != archive_expected:
        raise ValueError("Verified archive changed or lost members")
    tensors = [row for row in entry["files"] if row["uri"].endswith(".distcp")]
    if len(tensors) != 32 or not all(row["uri"].startswith(native + "policy/") for row in tensors):
        raise ValueError("Expected exactly the 32 original native policy tensor shards")
    deleted = {row["uri"] for row in tensors}
    retained = {uri: signature for uri, signature in expected.items() if uri not in deleted}
    current = members(complete["source_prefix"])
    if any(uri not in expected or signature != expected[uri] for uri, signature in current.items()):
        raise ValueError("Remaining source objects changed")
    if any(current.get(uri) != signature for uri, signature in retained.items()):
        raise ValueError("Required native metadata changed or disappeared")
    if not allow_removed and current != expected:
        raise ValueError("Original source inventory changed before the first removal")
    consumers = [(run["coordinator"], HUB), (run["gpu_job"], HUB)]
    consumers += [(run["export_job"], run.get("export_controller_config", HUB))]
    consumers += [(job_id, HUB) for job_id in run.get("failed_export_jobs", [])]
    consumers += [(submission["job_id"], submission["controller_config"])]
    stopped = [stopped_job(job_id, config) for job_id, config in consumers]
    if stopped[-1]["request_sha256"] != submission["request_sha256"]:
        raise ValueError("Stopped archive worker differs from the recorded request")
    model = verify_model(entry, run, manifest_path.parent)
    return {
        "entry": {key: value for key, value in entry.items() if key != "optimizer_resume_state_would_be_lost"},
        "complete_manifest_uri": complete_uri,
        "complete_manifest_sha256": complete_sha,
        "source_prefix": complete["source_prefix"],
        "archive_prefix": complete["archive_prefix"],
        "delete": [
            {**row, "archive": archived[row["uri"]]["archive"], "sha256": archived[row["uri"]]["sha256"]}
            for row in tensors
        ],
        "retain_metadata": {uri: list(signature) for uri, signature in retained.items()},
        "fresh_model_verification": model,
        "stopped_consumers": stopped,
        "delete_bytes": sum(row["bytes"] for row in tensors),
        "delete_objects": len(tensors),
        "native_restart_state_preserved": True,
    }


def plan(complete_uri, proposal_path, manifest_path, submission_path, output):
    complete, _ = content_manifest(complete_uri)
    (entry,) = [
        row
        for row in read(proposal_path)["entries"]
        if row["native_checkpoint"] == complete["source_prefix"] + "global_step_50/"
    ]
    result = evaluate(complete_uri, entry, manifest_path, read(submission_path))
    result.update(
        status="read-only plan; full native restart state retained in verified archive",
        inventory_proposal_sha256=hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
        archive_submission_sha256=hashlib.sha256(submission_path.read_bytes()).hexdigest(),
        source_deletions=0,
    )
    write(output, result)
    return {key: result[key] for key in ("status", "delete_objects", "delete_bytes")}


def remove(plan_path, manifest_path, submission_path, receipt):
    prepared = read(plan_path)
    if hashlib.sha256(submission_path.read_bytes()).hexdigest() != prepared["archive_submission_sha256"]:
        raise ValueError("Archive submission record changed after planning")
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if receipt.exists() and read(receipt)["plan_sha256"] != plan_sha:
        raise ValueError("Existing removal receipt belongs to another plan")
    current = evaluate(
        prepared["complete_manifest_uri"], prepared["entry"], manifest_path, read(submission_path), receipt.exists()
    )
    for field in ("complete_manifest_sha256", "source_prefix", "archive_prefix", "delete", "retain_metadata"):
        if current[field] != prepared[field]:
            raise ValueError(f"Removal differs from the read-only plan: {field}")
    intent = {"plan_sha256": plan_sha, **current}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    write(receipt, {**intent, "status": "removing redundant source tensors; full native archive retained"})
    observed = members(current["source_prefix"])
    targets = [row["uri"].removeprefix("s3://") for row in current["delete"] if row["uri"] in observed]
    if targets:
        fs, _ = filesystem_for(current["source_prefix"])
        fs.rm(targets)
    remaining = members(current["source_prefix"])
    if remaining != {uri: tuple(signature) for uri, signature in current["retain_metadata"].items()}:
        raise ValueError("Removal did not preserve exactly the required native metadata")
    write(receipt, {**intent, "status": "complete", "retained_metadata_objects": len(remaining)})
    return {"removed_redundant_bytes": current["delete_bytes"], "native_restart_state_preserved": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser("plan")
    preview.add_argument("complete_manifest_uri")
    preview.add_argument("proposal", type=Path)
    preview.add_argument("manifest", type=Path)
    preview.add_argument("submission", type=Path)
    preview.add_argument("output", type=Path)
    removal = commands.add_parser("remove")
    removal.add_argument("plan", type=Path)
    removal.add_argument("manifest", type=Path)
    removal.add_argument("submission", type=Path)
    removal.add_argument("receipt", type=Path)
    args = parser.parse_args()
    configure_coreweave_s3()
    if args.command == "plan":
        result = plan(args.complete_manifest_uri, args.proposal, args.manifest, args.submission, args.output)
    else:
        result = remove(args.plan, args.manifest, args.submission, args.receipt)
    print(json.dumps(result))
