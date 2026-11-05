# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the qualified probe on a regional host and preserve bounded attempt artifacts."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from google.cloud import storage

BASE = Path(__file__).parent
ROOT = BASE.parents[1]
ARTIFACTS = ("config.json", "report.json", "loss-rows.jsonl")


def validate_request(configs, config_sha256, index, checkpoint, counter, output_prefix):
    raw = configs.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config_sha256:
        raise ValueError("Published configuration file digest differs")
    row = json.loads(raw)[index]
    config = row.get("config", row)["train_config"]
    trainer = config["trainer"]
    if counter not in (3733, 18665, 33597):
        raise ValueError("Use a preregistered reference checkpoint")
    if (config["optimizer"]["alpha"], trainer["seed"], trainer["num_train_steps"]) != (0.25, 0, 37335):
        raise ValueError("Expected seed0 alpha=.25 full confirmation")
    expected = trainer["checkpointer"]["base_path"].rstrip("/") + f"/step-{counter - 1}"
    if checkpoint != expected or not checkpoint.startswith("gs://marin-us-central2/"):
        raise ValueError("Reference must be the exact permanent regional configuration checkpoint")
    if not output_prefix.startswith("gs://marin-us-central2/ema-gram-20-reference-probes/"):
        raise ValueError("Keep diagnostic artifacts in their dedicated regional prefix")
    qualification = json.loads((BASE / "reference-probe-svd-preflight.json").read_text())
    for path, digest in qualification["source_files"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Qualified probe source changed: {path}")
    if not all(run["success"] and run["state"] == "finished" for run in qualification["runs"]):
        raise ValueError("Probe qualification is incomplete")
    return config


def upload_artifact(bucket, prefix, relative, path):
    raw = path.read_bytes()
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError(f"Artifact exceeds the bounded diagnostic size: {relative}")
    blob = bucket.blob(prefix + "/" + relative)
    blob.upload_from_string(raw, if_generation_match=0, checksum="crc32c")
    blob.reload()
    if blob.download_as_bytes(if_generation_match=int(blob.generation), checksum="crc32c") != raw:
        raise ValueError(f"Artifact readback mismatch: {relative}")
    return dict(
        path=relative,
        generation=blob.generation,
        crc32c=blob.crc32c,
        size=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--configs-sha256", required=True)
    parser.add_argument("--config-index", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-counter", type=int, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    validate_request(
        args.configs, args.configs_sha256, args.config_index, args.checkpoint, args.expected_counter, args.output_prefix
    )
    identity = {
        "task": os.environ["IRIS_TASK_ID"],
        "attempt_uid": os.environ["IRIS_ATTEMPT_UID"],
        "source_commit": os.environ["GIT_COMMIT"],
        "configs_sha256": args.configs_sha256,
        "checkpoint": args.checkpoint,
        "expected_counter": args.expected_counter,
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    directory = args.output_dir / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    identity_path = directory / "attempt.json"
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    common = [
        sys.executable,
        str(BASE / "tracked_probe.py"),
        "--configs",
        str(args.configs),
        "--config-index",
        str(args.config_index),
        "--output-dir",
        str(directory),
    ]
    bucket_name, prefix = (args.output_prefix.rstrip("/") + "/" + directory.name)[5:].split("/", 1)
    bucket = storage.Client().bucket(bucket_name)
    outcome = {"success": False, "artifacts": [], **identity}
    try:
        subprocess.run([*common, "--mode", "logging-check"], check=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
        subprocess.run(
            [
                *common,
                "--mode",
                "probe",
                "--checkpoint",
                args.checkpoint,
                "--expected-counter",
                str(args.expected_counter),
            ],
            check=True,
        )
        outcome["success"] = True
    finally:
        # Preserve completed and partial bounded outputs, including failures. No
        # checkpoint arrays, credentials or unrestricted W&B directories are copied.
        paths = [identity_path, *(p for p in directory.glob("*/*") if p.name in ARTIFACTS)]
        for path in sorted(paths):
            outcome["artifacts"].append(upload_artifact(bucket, prefix, str(path.relative_to(directory)), path))
        manifest = directory / "upload-manifest.json"
        manifest.write_text(json.dumps(outcome, indent=2) + "\n")
        receipt = upload_artifact(bucket, prefix, manifest.name, manifest)
        print(
            json.dumps(
                {"output": "gs://" + bucket_name + "/" + prefix, "manifest": receipt, "success": outcome["success"]}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
