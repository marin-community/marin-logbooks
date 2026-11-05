# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute a frozen seed0 confirmation through the existing native training path."""

import argparse
import hashlib
import json
import os
import uuid
from dataclasses import replace
from pathlib import Path

import draccus
from levanter.main.train_lm import TrainLmConfig
from levanter.tracker.wandb import WandbConfig
from prepare_subset_confirmations import prepare

from experiments.ema_gram.plan import read_lock
from experiments.ema_gram.subset_campaign import prepared
from experiments.ema_gram.train import run_training


def resolve(bundle_path, lock_path, index):
    bundle = json.loads(bundle_path.read_text())
    expected = prepare(read_lock(lock_path), bundle["artifact_version"])
    if bundle != expected:
        raise ValueError("Confirmation bundle differs from the qualified locked configuration")
    row = bundle["cells"][index]
    encoded = row["config"]
    config = draccus.decode(TrainLmConfig, encoded["train_config"])
    # Draccus treats bool values as strings for this str|bool mapping. Restore
    # the frozen JSON types, then require a complete round-trip identity check.
    config = replace(
        config, trainer=replace(config.trainer, jax_config=encoded["train_config"]["trainer"]["jax_config"])
    )
    pod = replace(
        prepared(0, row["variant"]),
        train_config=config,
        output_path=encoded["output_path"],
        env_vars=encoded["env_vars"],
        auto_build_caches=False,
    )
    if draccus.encode(pod) != encoded:
        raise ValueError("Resolved native confirmation config differs from the frozen bundle")
    return pod, row, bundle["selection_lock_digest"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--mode", choices=("logging-check", "worker"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pod, row, digest = resolve(args.bundle, args.lock, args.index)
    record = dict(
        config=row["config"],
        config_sha256=row["config_sha256"],
        selection_lock_digest=digest,
        bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        source_commit=os.environ.get("GIT_COMMIT"),
        task=os.environ.get("IRIS_TASK_ID"),
        attempt_uid=os.environ.get("IRIS_ATTEMPT_UID"),
        entrypoint_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    if args.mode == "worker":
        run_training(pod)
        record["native_returned"] = True
    else:
        identity = "ema-gram-confirmation-host-" + uuid.uuid4().hex[:12]
        tracker = WandbConfig(
            entity="marin-community",
            project="marin",
            id=identity,
            name=identity,
            group="ema-gram-20-confirmations",
            mode="online",
            resume="never",
            background=False,
            save_code=False,
        ).init(None)
        print(f"W&B: https://wandb.ai/marin-community/marin/runs/{identity}", flush=True)
        try:
            tracker.log_hyperparameters(record)
            tracker.log({"logging_check/success": 1}, step=0)
            record.update(identity=identity, success=True)
        finally:
            tracker.finish()
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
