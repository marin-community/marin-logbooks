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
from prepare_parallel_long_runs import resolve_bundle
from prepare_subset_confirmations import prepare

from experiments.ema_gram.plan import read_lock
from experiments.ema_gram.subset_campaign import prepared
from experiments.ema_gram.train import run_training


def resolve(bundle_path, lock_path, index, amendment_path=None, completions_path=None):
    if amendment_path is not None:
        bundle = resolve_bundle(bundle_path, completions_path, amendment_path)
        digest = bundle["amendment_digest"]
    else:
        bundle = json.loads(bundle_path.read_text())
        expected = prepare(read_lock(lock_path), bundle["artifact_version"])
        if bundle != expected:
            raise ValueError("Confirmation bundle differs from the qualified locked configuration")
        digest = bundle["selection_lock_digest"]
    row = bundle["cells"][index]
    encoded = row["config"]
    config = draccus.decode(TrainLmConfig, encoded["train_config"])
    # Preserve union field types across Draccus fresh-process decoding, then
    # require complete encoded-config identity before invoking the trainer.
    config = replace(
        config,
        trainer=replace(
            config.trainer,
            jax_config=encoded["train_config"]["trainer"]["jax_config"],
            load_checkpoint_path=encoded["train_config"]["trainer"]["load_checkpoint_path"],
        ),
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
    return pod, row, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--lock", type=Path)
    selection.add_argument("--amendment", type=Path)
    parser.add_argument("--completions", type=Path)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--mode", choices=("logging-check", "worker"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pod, row, digest = resolve(args.bundle, args.lock, args.index, args.amendment, args.completions)
    record = dict(
        config=row["config"],
        config_sha256=row["config_sha256"],
        selection_lock_digest=digest if args.amendment is None else None,
        amendment_digest=digest if args.amendment is not None else None,
        global_selection_locked=args.amendment is None,
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
