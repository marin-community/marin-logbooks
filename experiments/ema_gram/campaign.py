# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one full-schedule campaign cell through the existing native trainer."""

import argparse
import hashlib
import json
import uuid
from dataclasses import replace
from pathlib import Path

import draccus
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import materialized_config
from marin.training.training import apply_output_path

from experiments.ema_gram.plan import Cell
from experiments.ema_gram.prefix_train import SOURCE_PATHS, cpu_check
from experiments.ema_gram.train import build_training_step, run_training

BASE = Path("logbooks/20-ema-gram-polar")
VERSION = "2026.11.04.02"


def source_hashes():
    return {
        p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (*SOURCE_PATHS, "experiments/ema_gram/campaign.py")
    }


def prepared(index, variant):
    cell = Cell(**json.loads((BASE / "single-seed-pilots.json").read_text())[index])
    if variant not in ("v4-8", "v4-32"):
        raise ValueError("Only established full-training topologies are enabled")
    step = build_training_step(
        cell,
        VERSION,
        tpu_type=variant,
        host_memory="256g",
        eigh_layout="matrix",
        eigh_implementation="default",
        initialize_from=None,
        profile_start_step=100,
        profile_steps=1,
    )
    pod = materialized_config(step, "gs://marin-us-central2")
    return replace(pod, train_config=apply_output_path(pod.train_config, pod.output_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "logging-check", "cpu-check", "worker"), required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--variant", choices=("v4-8", "v4-32"), default="v4-32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.index < 29:
        raise ValueError("Invalid initial-grid cell index")
    pod = prepared(args.index, args.variant)
    resolved = draccus.encode(pod)
    record = dict(source_files=source_hashes(), index=args.index, variant=args.variant, config=resolved)
    if args.mode == "prepare":
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        return
    manifest = json.loads((BASE / "single-seed-preflight.json").read_text())
    if manifest["source_files"] != source_hashes():
        raise ValueError("Campaign numerical source differs from the preflight manifest")
    if args.mode == "worker":
        # Existing native entrypoint owns the full schedule and all checkpoint restore semantics.
        run_training(pod)
        args.output.write_text(json.dumps(dict(record, native_returned=True), indent=2) + "\n")
        return
    identity = f"ema-gram-campaign-{args.mode}-{uuid.uuid4().hex[:12]}"
    tracker = WandbConfig(
        entity="marin-community",
        project="marin",
        id=identity,
        name=identity,
        group=f"ema-gram-{VERSION}",
        mode="online",
        resume="never",
        background=False,
        save_code=False,
    ).init(None)
    print(f"W&B: https://wandb.ai/marin-community/marin/runs/{identity}", flush=True)
    try:
        tracker.log_hyperparameters(record)
        tracker.log({"logging_check/success": 1}, step=0)
        if args.mode == "cpu-check":
            # Reuse the native training/checkpoint harness; no production training loop is reimplemented.
            cpu_check(tracker, args.output)
            tracker.log({"campaign/cpu_success": 1}, step=100)
        else:
            args.output.write_text(json.dumps(dict(record, identity=identity, success=True), indent=2) + "\n")
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
