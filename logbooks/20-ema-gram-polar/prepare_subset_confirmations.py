# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize seed0 confirmation configs; this tool cannot submit training jobs."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import draccus
from marin.execution.lazy import materialized_config
from marin.training.training import apply_output_path

from experiments.ema_gram.plan import confirmations, read_lock
from experiments.ema_gram.subset_campaign import BASE, prepared, source_hashes
from experiments.ema_gram.train import build_training_step

REFERENCE_COUNTERS = (3733, 18665, 33597)
REFERENCE_KEEP = [{"every": counter - 1, "until": counter - 1} for counter in REFERENCE_COUNTERS]


def prepare(lock, version):
    if not version or version == "2026.11.04.03" or "/" in version:
        raise ValueError("Use a distinct confirmation artifact version")
    validation = json.loads((BASE / "subset-training-validation.json").read_text())
    hashes = source_hashes()
    if not validation["passed"] or validation["source_files"] != hashes:
        raise ValueError("Training numerical sources no longer match existing validation")
    for name in ("subset-inputs.json", "subset-dataset.json", "subset-cache-manifest.json"):
        if validation["artifacts"][name] != hashlib.sha256((BASE / name).read_bytes()).hexdigest():
            raise ValueError(f"Frozen subset artifact changed: {name}")
    cells = [cell for cell in confirmations(lock) if cell.seed == 0]
    if len(cells) not in (3, 4) or not {0, 0.25, 0.5}.issubset({cell.alpha for cell in cells}):
        raise ValueError("Expected three or four seed0 confirmations")
    result = []
    for cell in cells:
        variant = "v4-8" if cell.alpha == 0 else "v4-32"
        subset_data = prepared(0, variant).train_config.data
        step = build_training_step(
            cell,
            version,
            tpu_type=variant,
            host_memory="256g",
            eigh_layout="matrix",
            eigh_implementation="default",
            initialize_from=None,
            profile_start_step=100,
            profile_steps=1,
        )
        pod = materialized_config(step, "gs://marin-us-central2")
        config = apply_output_path(pod.train_config, pod.output_path)
        config = replace(config, data=subset_data)
        if cell.alpha == 0.25:
            config = replace(
                config,
                trainer=replace(config.trainer, checkpointer=replace(config.trainer.checkpointer, keep=REFERENCE_KEEP)),
            )
        if config.trainer.num_train_steps != 37335 or config.trainer.train_batch_size != 8:
            raise ValueError("Confirmation schedule or global batch changed")
        if config.trainer.initialize_from is not None or config.initialize_from_checkpoint_path is not None:
            raise ValueError("Confirmations must start from scratch")
        if config.trainer.seed != 0 or config.data_seed != 0 or config.trainer.tracker.mode != "online":
            raise ValueError("Confirmations require seed0 and native ONLINE tracking")
        if config.optimizer.eigh_layout != "matrix" or config.optimizer.eigh_implementation != "default":
            raise ValueError("Preserve the validated matrix/default optimizer")
        pod = replace(pod, train_config=config)
        encoded = draccus.encode(pod)
        result.append(
            dict(
                cell=asdict(cell),
                variant=variant,
                config=encoded,
                config_sha256=hashlib.sha256(json.dumps(encoded, sort_keys=True).encode()).hexdigest(),
                reference_checkpoint_counters=list(REFERENCE_COUNTERS) if cell.alpha == 0.25 else [],
                reference_checkpoint_labels=(
                    [counter - 1 for counter in REFERENCE_COUNTERS] if cell.alpha == 0.25 else []
                ),
                final_checkpoint_label=37334,
            )
        )
    return dict(
        selection_lock_digest=lock.sha256,
        artifact_version=version,
        source_files=hashes,
        source_numerical_validation_reused=True,
        cells=result,
        submitted=False,
        launch_ready=False,
        remaining_launch_gates=[
            "Independently validated complete pilot/expansion results and published scientific selection lock",
            "Exact-source fresh campaign worktree, separate ledger and job identities",
            "Full catalog check and fresh execution-host ONLINE logging before native trainer",
            "One monitoring owner for accepted confirmation identities",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(read_lock(args.lock), args.version)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
