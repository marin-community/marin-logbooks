# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind every seed-zero pilot to one immutable regional subset and native trainer."""

import argparse
import hashlib
import json
import math
import uuid
from dataclasses import replace
from pathlib import Path

import draccus
import jax
import numpy as np
from levanter.tracker.wandb import WandbConfig
from levanter.utils.thread_utils import blocking_wait
from marin.execution.lazy import materialized_config
from marin.training.training import apply_output_path

from experiments.ema_gram.plan import pilots
from experiments.ema_gram.prefix_train import SOURCE_PATHS, cpu_check
from experiments.ema_gram.train import build_training_step, run_training

BASE = Path("logbooks/20-ema-gram-polar")
VERSION = "2026.11.04.03"
CELLS = pilots([0.01, 0.005, 0.002, 0.001, 0.0005])


def source_hashes():
    paths = (*SOURCE_PATHS, "experiments/ema_gram/subset_campaign.py")
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def prepared(index, variant):
    if variant not in ("v4-8", "v4-32") or not 0 <= index < len(CELLS):
        raise ValueError("Invalid cell or unqualified training topology")
    inputs = json.loads((BASE / "subset-inputs.json").read_text())
    step = build_training_step(
        CELLS[index],
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
    config = apply_output_path(pod.train_config, pod.output_path)
    components = dict(config.data.components)
    names = {task["component"] for task in inputs["tasks"]}
    assert names == set(inputs["train_weights"])
    web_template = components["nemotron_cc/hq_actual"]
    components = {k: v for k, v in components.items() if not k.startswith("nemotron_cc/")}
    for name in names:
        component = components.get(name, web_template)
        cache = inputs["output_prefix"] + "/" + name
        # A single regional dataset is shared by all cells. Auto-building is disabled.
        components[name] = replace(component, cache_dir=cache, source=replace(component.source, cache_dir=cache))
    weights = {k: inputs["train_weights"].get(k, 0.0) for k in components}
    config = replace(config, data=replace(config.data, components=components, train_weights=weights))
    return replace(pod, train_config=config)


def check_data(config):
    data = config.data
    Pos = config.model.max_Pos
    assert Pos.size == 4096 and config.data_seed == 0
    caches = data.build_caches("train")
    samples = {}
    for name, cache in caches.items():
        rows = len(cache)
        edges = cache.get_batch_sync([0, rows - 1])
        samples[name] = dict(
            rows=rows,
            tokens=cache.flat_field_length("input_ids"),
            edge_sha256=[hashlib.sha256(np.asarray(r["input_ids"], dtype="<i4").tobytes()).hexdigest() for r in edges],
        )
    mixture = data.train_mixture(Pos, config.trainer.batch_schedule, key=jax.random.PRNGKey(0))
    # Count through the next complete native mixture block; this bounds every
    # component's consumption across the entire later confirmation schedule.
    needed = mixture.sequence_counts_before_block(math.ceil(37335 * 8 / mixture.block_size))
    capacities = {name: blocking_wait(dataset.async_len()) for name, dataset in mixture.datasets.items()}
    assert all(capacities[name] >= count for name, count in needed.items()), (capacities, needed)
    indices = list(range(8)) + list(range(7630 * 8 - 8, 7630 * 8)) + list(range(37335 * 8 - 8, 37335 * 8))
    rows = blocking_wait(mixture.get_batch(indices))
    hashes = [hashlib.sha256(np.asarray(row.tokens, dtype="<i4").tobytes()).hexdigest() for row in rows]
    # Rebuild the native data graph and demand the same seed-zero token sequence.
    replay = data.train_mixture(Pos, config.trainer.batch_schedule, key=jax.random.PRNGKey(0))
    again = blocking_wait(replay.get_batch(indices))
    assert hashes == [hashlib.sha256(np.asarray(row.tokens, dtype="<i4").tobytes()).hexdigest() for row in again]
    evaluations = data.validation_sets(Pos)
    eval_counts = {}
    for name, dataset in evaluations.items():
        sample = blocking_wait(dataset.get_batch([0]))
        assert len(sample) == 1
        eval_counts[name] = blocking_wait(dataset.async_len())
    assert len(samples) == 3 and len(eval_counts) == 23
    return dict(
        passed=True,
        training_components=samples,
        confirmation_required_sequences=needed,
        available_sequences=capacities,
        mixture_indices=indices,
        mixture_sha256=hashes,
        deterministic_replay=True,
        eval_sequences=eval_counts,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("prepare", "logging-check", "cpu-check", "data-check", "worker"), required=True
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--variant", choices=("v4-8", "v4-32"), default="v4-32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pod = prepared(args.index, args.variant)
    record = dict(source_files=source_hashes(), index=args.index, variant=args.variant, config=draccus.encode(pod))
    if args.mode == "prepare":
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        return
    if args.mode == "worker":
        validated = json.loads((BASE / "subset-training-validation.json").read_text())
        assert validated["passed"] and validated["source_files"] == source_hashes()
        assert validated["dataset_sha256"] == hashlib.sha256((BASE / "subset-dataset.json").read_bytes()).hexdigest()
        run_training(pod)
        args.output.write_text(json.dumps(dict(record, native_returned=True), indent=2) + "\n")
        return
    identity = f"ema-gram-subset-training-{args.mode}-{uuid.uuid4().hex[:12]}"
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
        record["identity"] = identity
        if args.mode == "cpu-check":
            cpu_check(tracker, args.output)
            record["checks"] = json.loads(args.output.read_text())
        elif args.mode == "data-check":
            record["checks"] = check_data(pod.train_config)
        record["success"] = True
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        tracker.log({"subset_training/success": 1}, step=100)
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
