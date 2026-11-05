# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute a frozen seed0 pilot expansion through the existing native training path."""

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
from prepare_subset_expansions import prepare

from experiments.ema_gram.subset_campaign import prepared
from experiments.ema_gram.train import run_training


def resolve(bundle_path, completions_path, accepted_path, index):
    bundle = json.loads(bundle_path.read_text())
    records = json.loads(completions_path.read_text())
    accepted = json.loads(accepted_path.read_text())
    expected = prepare(records, bundle["artifact_version"], accepted)
    if bundle != expected:
        raise ValueError("Expansion bundle differs from completed-group configuration")
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
    # Draccus also promotes the four frozen integer-valued RoPE fields to
    # floats. Restore their native input representation before digest checking.
    frozen_rope = encoded["train_config"]["model"]["rope"]
    config = replace(
        config,
        model=replace(
            config.model,
            rope=replace(
                config.model.rope,
                **{key: frozen_rope[key] for key in ("theta", "factor", "low_freq_factor", "high_freq_factor")},
            ),
        ),
    )
    pod = replace(
        prepared(0, row["variant"]),
        train_config=config,
        output_path=encoded["output_path"],
        env_vars=encoded["env_vars"],
        auto_build_caches=False,
    )
    if (
        draccus.encode(pod) != encoded
        or hashlib.sha256(json.dumps(draccus.encode(pod), sort_keys=True).encode()).hexdigest() != row["config_sha256"]
    ):
        raise ValueError("Resolved native expansion config differs from the frozen bundle")
    return pod, row, bundle["completion_records_sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--completions", type=Path, required=True)
    parser.add_argument("--accepted-expansions", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--mode", choices=("logging-check", "worker"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pod, row, digest = resolve(args.bundle, args.completions, args.accepted_expansions, args.index)
    record = dict(
        config=row["config"],
        config_sha256=row["config_sha256"],
        completion_records_sha256=digest,
        accepted_expansions_sha256=hashlib.sha256(args.accepted_expansions.read_bytes()).hexdigest(),
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
        identity = "ema-gram-expansion-host-" + uuid.uuid4().hex[:12]
        tracker = WandbConfig(
            entity="marin-community",
            project="marin",
            id=identity,
            name=identity,
            group="ema-gram-20-expansions",
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
