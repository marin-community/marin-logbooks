# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a paired EMA-Gram continuation without launching or initializing arrays."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import draccus
from marin.execution.lazy import materialized_config
from marin.training.training import apply_output_path

from experiments.ema_gram.plan import Cell
from experiments.ema_gram.train import build_training_step


def differences(left, right, path=()):
    if isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        return [row for key in left for row in differences(left[key], right[key], (*path, key))]
    if left != right:
        return [dict(path=list(path), original=left, candidate=right)]
    return []


def prepare(checkpoint, versions, host_memory="256g"):
    assert len(versions) == 2 and versions[0] != versions[1]
    cell = Cell(alpha=0.25, ridge=1e-4, learning_rate=0.002, seed=0)
    configs = {}
    for layout, version in zip(("replicated", "matrix"), versions, strict=True):
        step = build_training_step(
            cell,
            version,
            tpu_type="v4-32",
            host_memory=host_memory,
            eigh_layout=layout,
            eigh_implementation="default",
            initialize_from=checkpoint,
            profile_start_step=940,
            profile_steps=1,
        )
        pod = materialized_config(step, "gs://marin-us-central2")
        configs[layout] = draccus.encode(replace(pod, train_config=apply_output_path(pod.train_config, pod.output_path)))
    changed = differences(configs["replicated"], configs["matrix"])
    allowed = {
        ("train_config", "optimizer", "eigh_layout"),
        ("train_config", "trainer", "id"),
        ("train_config", "trainer", "tracker", "id"),
        ("train_config", "trainer", "tracker", "name"),
        ("train_config", "trainer", "tracker", "group"),
        ("train_config", "trainer", "tracker", "replicate_path"),
        ("train_config", "trainer", "checkpointer", "base_path"),
        ("train_config", "trainer", "checkpointer", "temporary_base_path"),
        ("output_path",),
        ("train_config", "hf_save_path"),
        ("train_config", "trainer", "load_checkpoint_path"),
    }
    assert {tuple(row["path"]) for row in changed} <= allowed
    assert any(row["path"] == ["train_config", "optimizer", "eigh_layout"] for row in changed)
    for cfg in configs.values():
        t = cfg["train_config"]["trainer"]
        assert t["num_train_steps"] == 7630 and t["train_batch_size"] == 8
        assert t["seed"] == cfg["train_config"]["data_seed"] == 0
        assert t["initialize_from"] == checkpoint and t["load_checkpoint"] is None
        assert not t["allow_partial_checkpoint"]
        assert t["tracker"]["mode"] == "online" and t["tracker"]["resume"] == "never"
        assert t["mesh"]["axes"] == dict(data=4, model=4, replica=1)
        assert cfg["resources"]["preemptible"] and cfg["resources"]["regions"] == ["us-central2"]
        assert cfg["resources"]["replicas"] == 4
    source_paths = [
        "experiments/ema_gram/train.py",
        "experiments/ema_gram/plan.py",
        "experiments/references/train_delphi_seeds.py",
        "lib/marin/src/marin/training/training.py",
        "lib/levanter/src/levanter/optim/ema_gram.py",
        "lib/levanter/src/levanter/trainer.py",
        "lib/levanter/src/levanter/main/train_lm.py",
    ]
    return dict(
        state="Prepared only; no training submitted and no full checkpoint restore claimed",
        checkpoint=checkpoint,
        configs=configs,
        allowed_differences=changed,
        source_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in source_paths},
        comparison=dict(
            primary="Training and held-out loss at matched steps; quantify gap and variability",
            performance="Measured steady throughput, absolute MFU and wall-clock progress at comparable loss",
            diagnostic="rtol=atol=1e-5 retained; failed update equivalence alone is not a training veto",
            no_loss_equivalence_margin_declared=True,
        ),
        remaining=[
            "Verify full regional trainer restore, restored step and data position before continuation.",
            "Validate full-trainer host/device memory; this config does not resolve the earlier synthetic v4-32 OOM.",
            "Run online logging check on execution host; new numerical entrypoints need own online CPU validation.",
            "Publish exact source before execution, run original/candidate sequentially and preserve all attempts.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host-memory", default="256g")
    parser.add_argument("--versions", nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.checkpoint, args.versions, args.host_memory)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
