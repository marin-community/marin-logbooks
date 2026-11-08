# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run ONLINE checks and supplemental probes, preserving each attempt's bounded outputs."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import wandb
from google.cloud import storage
from levanter.tracker.wandb import WandbTracker
from rigging.secrets import require_secret
from run_reference_probe import upload_artifact

ROOT = Path(__file__).resolve().parents[2]


def worker(mode, settings, config, output):
    identity = "ema-gram-supplement-" + mode + "-" + uuid.uuid4().hex[:12]
    destination = output / identity
    destination.mkdir()
    run = wandb.init(
        entity="marin-community",
        project=settings["wandb_project"],
        id=identity,
        name=identity,
        group=settings["campaign"],
        mode="online",
        resume="never",
        settings={"api_key": require_secret("WANDB_API_KEY")},
    )
    if run.settings.mode != "online":
        raise ValueError("ONLINE tracking is required")
    tracker = WandbTracker(run, replicate_path=str(destination))
    metadata = dict(
        settings=settings,
        resolved_config=config,
        source_commit=os.environ.get("GIT_COMMIT"),
        task=os.environ.get("IRIS_TASK_ID"),
        attempt_uid=os.environ.get("IRIS_ATTEMPT_UID"),
        mode=mode,
        wandb_url=run.url,
        identity=identity,
    )
    (destination / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    tracker.log_hyperparameters(metadata)
    print(f"W&B: {run.url}", flush=True)
    record_count = 0
    with (destination / "rows.jsonl").open("x") as stream:

        def emit(row):
            nonlocal record_count
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            record_count += 1
            metrics = {"supplement/records": record_count, "supplement/kind": row["kind"]}
            for key in ("counter", "alpha", "step_size", "delta_mean", "delta_sem"):
                if key in row:
                    metrics["supplement/" + key] = row[key]
            tracker.log(metrics, step=record_count)
            print(json.dumps({k: v for k, v in row.items() if k in ("kind", "counter", "alpha", "method")}), flush=True)

        try:
            tracker.log({"logging_check/success": 1}, step=0)
            if mode == "oracle":
                from check_supplemental_probe import execute  # noqa: PLC0415 - ONLINE precedes kernels

                emit(dict(kind="oracle", **execute()))
            elif mode == "probe":
                from iris.jax.init import initialize_jax  # noqa: PLC0415

                initialize_jax()
                from supplemental_probe import run as run_probe  # noqa: PLC0415

                run_probe(settings, config, emit)
            run.summary["supplement/success"] = 1
        except BaseException:
            run.finish(exit_code=1)
            raise
        else:
            tracker.finish()
            (destination / "completion.json").write_text(
                json.dumps(dict(success=True, records=record_count, wandb_url=run.url)) + "\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("validate", "execute", "logging-check", "oracle", "probe"), default="validate"
    )
    args = parser.parse_args()
    raw = args.request.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.request_sha256:
        raise ValueError("Supplement request hash mismatch")
    settings = json.loads(raw)
    for name, digest in settings["source_files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Published source differs: {name}")
    configs = (ROOT / settings["configs"]).read_bytes()
    if hashlib.sha256(configs).hexdigest() != settings["configs_sha256"]:
        raise ValueError("Frozen configuration differs")
    config = json.loads(configs)[0]["config"]["train_config"]
    base = config["trainer"]["checkpointer"]["base_path"].rstrip("/")
    for item in settings["checkpoints"]:
        if item["checkpoint"] != base + f"/step-{item['counter'] - 1}":
            raise ValueError("Unexpected reference checkpoint")
    if args.mode == "validate":
        print(json.dumps(dict(validated=True, submitted=False, campaign=settings["campaign"])))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode != "execute":
        worker(args.mode, settings, config, args.output)
        return
    directory = args.output / uuid.uuid4().hex
    directory.mkdir()
    identity = dict(
        task=os.environ["IRIS_TASK_ID"],
        attempt_uid=os.environ["IRIS_ATTEMPT_UID"],
        source_commit=os.environ["GIT_COMMIT"],
        request_sha256=args.request_sha256,
    )
    (directory / "attempt.json").write_text(json.dumps(identity, indent=2) + "\n")
    common = [
        sys.executable,
        __file__,
        "--request",
        str(args.request),
        "--request-sha256",
        args.request_sha256,
        "--output",
        str(directory),
    ]
    outcome = dict(success=False, artifacts=[], **identity)
    try:
        subprocess.run([*common, "--mode", "logging-check"], check=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
        subprocess.run([*common, "--mode", "probe"], check=True)
        outcome["success"] = True
    finally:
        bucket_name, prefix = (settings["output_prefix"] + "/" + directory.name)[5:].split("/", 1)
        bucket = storage.Client().bucket(bucket_name)
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".jsonl") and "wandb" not in path.parts:
                relative = str(path.relative_to(directory))
                outcome["artifacts"].append(upload_artifact(bucket, prefix, relative, path))
        manifest = directory / "upload-manifest.json"
        manifest.write_text(json.dumps(outcome, indent=2) + "\n")
        receipt = upload_artifact(bucket, prefix, manifest.name, manifest)
        print(
            json.dumps(dict(output="gs://" + bucket_name + "/" + prefix, manifest=receipt, success=outcome["success"])),
            flush=True,
        )


if __name__ == "__main__":
    main()
