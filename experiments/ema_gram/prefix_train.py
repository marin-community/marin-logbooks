# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the first requested updates with the original optimizer scheduling horizon."""

import argparse
import hashlib
import itertools
import json
import math
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import draccus
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from levanter.checkpoint import save_checkpoint
from levanter.models.qwen import Qwen3Config
from levanter.tracker import current_tracker
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import Trainer
from levanter.trainer_state import TrainerState
from marin.execution.lazy import materialized_config
from marin.training.training import apply_output_path
from rigging.filesystem.storage_path import StoragePath

from experiments.ema_gram.plan import Cell
from experiments.ema_gram.restore import audit_state, describe_arrays
from experiments.ema_gram.train import build_training_step, run_training

SOURCE_PATHS = (
    "experiments/ema_gram/prefix_train.py",
    "experiments/ema_gram/train.py",
    "experiments/ema_gram/plan.py",
    "experiments/references/train_delphi_seeds.py",
    "lib/marin/src/marin/training/training.py",
    "lib/levanter/src/levanter/optim/ema_gram.py",
    "lib/levanter/src/levanter/trainer.py",
    "lib/levanter/src/levanter/main/train_lm.py",
    "experiments/ema_gram/restore.py",
)


def source_hashes():
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCE_PATHS}


def prepared_pod(version):
    step = build_training_step(
        Cell(0.25, 1e-4, 0.002, seed=0),
        version,
        tpu_type="v4-32",
        host_memory="256g",
        eigh_layout="matrix",
        eigh_implementation="default",
        initialize_from=None,
        profile_start_step=100,
        profile_steps=1,
    )
    pod = materialized_config(step, "gs://marin-us-central2")
    return replace(pod, train_config=apply_output_path(pod.train_config, pod.output_path))


def train_to_counter(native_train, trainer, state, loader, target):
    """Bound data consumption without rebuilding or changing the optimizer schedule."""
    start = int(state.step)
    if not 0 <= start <= target <= trainer.config.num_train_steps:
        raise ValueError("Invalid restored counter or prefix target")
    if start == target:
        # Do not invent a zero loss or rerun an update on an already-complete retry.
        # Its prior final-hook evidence must be checked independently.
        return None
    result = native_train(trainer, state, itertools.islice(loader, target - start))
    assert int(result.state.step) == target and math.isfinite(float(result.loss))
    return result


def run_prefix(pod, target):
    native_train = Trainer.train
    identity = uuid.uuid4().hex[:12]
    reports = StoragePath(pod.output_path) / "prefix-reports" / identity

    def bounded_train(trainer, state, loader):
        tracker = current_tracker()
        start = int(state.step)
        report = dict(
            process_index=jax.process_index(),
            process_count=jax.process_count(),
            start_counter=start,
            target_counter=target,
            optimizer_schedule_counter=trainer.config.num_train_steps,
            initialize_from=trainer.config.initialize_from,
            checkpoint_load_plan=trainer.checkpoint_load_plan(),
            source_sha256=source_hashes(),
            data_start_counter=loader._start_from_batch,
            global_example_offset=loader.dl.scheduler.global_data_offset_by_step(start),
            completed=False,
        )
        assert trainer.config.initialize_from is None and trainer.config.num_train_steps == 7630
        assert loader._start_from_batch == start
        out = reports / f"process-{jax.process_index()}.json"
        out.write_text(json.dumps(report, indent=2) + "\n")
        tracker.log_hyperparameters(dict(prefix_observation=report))
        tracker.log({"prefix/start_counter": start, "prefix/target_counter": target}, step=start)
        result = train_to_counter(native_train, trainer, state, loader, target)
        report.update(
            completed=result is not None,
            final_counter=target,
            updates_in_this_process=target - start,
            final_hooks_executed=result is not None,
            already_at_endpoint=result is None,
            final_loss=float(result.loss) if result is not None else None,
            scope="Native final hooks returned; require durable checkpoint, eval and canonical success separately",
        )
        out.write_text(json.dumps(report, indent=2) + "\n")
        tracker.log(
            {"prefix/completed": int(result is not None), "prefix/already_at_endpoint": int(result is None)}, step=target
        )
        return result

    with patch.object(Trainer, "train", bounded_train):
        return run_training(pod)


def cpu_check(tracker, output):
    """Compare the bounded native loop with an uninterrupted reference and restored suffix."""
    assert jax.default_backend() == "cpu"
    pod = prepared_pod("prefix-cpu-dev")
    model_cfg = Qwen3Config(
        hidden_dim=8, intermediate_dim=16, num_layers=2, num_heads=2, num_kv_heads=2, head_dim=4, max_seq_len=8
    )
    tx = pod.train_config.optimizer.build(7630)
    initial = TrainerState.init(
        tx, model_cfg.build(hax.Axis("vocab", 16), key=jax.random.PRNGKey(0)), key=jax.random.PRNGKey(1)
    )
    before = describe_arrays(initial.saveable_state)

    def fresh_state():
        # Native Trainer donates its input buffers; each reference needs independent storage.
        return jax.tree.map(lambda x: jnp.array(x, copy=True) if isinstance(x, jax.Array) else x, initial)

    def loss(model, batch, *, key=None):
        del key
        return sum(jnp.mean(x**2) for x in jax.tree.leaves(model) if isinstance(x, jax.Array)) * jnp.mean(batch.array)

    cfg = replace(
        pod.train_config.trainer,
        id="prefix-cpu",
        train_batch_size=1,
        require_accelerator=False,
        mesh=replace(pod.train_config.trainer.mesh, axes={"data": 1, "model": 1, "replica": 1}),
    )
    with tempfile.TemporaryDirectory(prefix="ema-prefix-") as folder:
        cfg = replace(cfg, checkpointer=replace(cfg.checkpointer, base_path=folder, temporary_base_path=None))
        with current_tracker(tracker), Trainer(cfg, tx, loss, add_default_hooks=False) as trainer:
            records = []

            def record(info, force=False):
                records.append((info.step, float(info.loss), force))
                tracker.log({"prefix_cpu/loss": float(info.loss)}, step=len(records))

            trainer.add_hook(record, every=1)
            batches = [hax.named(jnp.asarray([x], dtype=jnp.float32), hax.Axis("batch", 1)) for x in (1, 2, 3, 4, 5, 6)]
            result = train_to_counter(Trainer.train, trainer, fresh_state(), iter(batches), 4)
            assert result is not None and int(result.state.step) == 4
            first_records = list(records)
            assert [r[0] for r in records] == [0, 1, 2, 3, 3] and records[-1][2]
            oracle = fresh_state()
            for batch in batches[:4]:
                oracle = trainer.train_step(oracle, batch).state
            assert describe_arrays(result.state.saveable_state) == describe_arrays(oracle.saveable_state)
            partial = train_to_counter(Trainer.train, trainer, fresh_state(), iter(batches), 2)
            assert partial is not None
            checkpoint = str(Path(folder) / "step-1")
            save_checkpoint(partial.state.saveable_state, 1, checkpoint)
            trainer.config = replace(cfg, initialize_from=checkpoint, load_checkpoint=False)
            loaded = trainer.initial_state(
                jax.random.PRNGKey(99),
                model_init=lambda: model_cfg.build(hax.Axis("vocab", 16), key=jax.random.PRNGKey(9)),
            )
            restored = audit_state(loaded, checkpoint, 2)
            resumed = train_to_counter(Trainer.train, trainer, loaded, iter(batches[2:]), 4)
            assert resumed is not None
            assert describe_arrays(resumed.state.saveable_state) == describe_arrays(oracle.saveable_state)
            untouched_count = len(records)
            assert train_to_counter(Trainer.train, trainer, resumed.state, iter(batches), 4) is None
            assert len(records) == untouched_count
            assert trainer.config.num_train_steps == 7630 and describe_arrays(initial.saveable_state) == before
            assert all(np.isfinite(r[1]) for r in records)
            report = dict(
                exact_uninterrupted_state_match=True,
                exact_restored_suffix_match=True,
                initial_state_immutable=True,
                source_sha256=source_hashes(),
                labels_and_forced_hooks=first_records,
                schedule_counter=trainer.config.num_train_steps,
                completed_retry_has_no_update_or_fake_loss=True,
                restored_checkpoint=restored,
                scope="Tiny online CPU native Trainer loop; not real-model TPU proof",
            )
            output.write_text(json.dumps(report, indent=2) + "\n")
            tracker.log({"prefix_cpu/success": 1}, step=len(records) + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "logging-check", "cpu-check", "worker"), required=True)
    parser.add_argument("--version", default="2026.11.04.01")
    parser.add_argument("--target-counter", type=int, default=924)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pod = prepared_pod(args.version)
    resolved = draccus.encode(pod)
    if args.mode == "prepare":
        args.output.write_text(
            json.dumps(
                dict(config=resolved, source_sha256=source_hashes(), target_counter=args.target_counter), indent=2
            )
            + "\n"
        )
        return None
    if args.manifest is not None:
        expected = json.loads(args.manifest.read_text())
        assert expected["config"] == resolved and expected["source_sha256"] == source_hashes()
        assert expected["target_counter"] == args.target_counter
    if args.mode == "worker":
        assert args.manifest is not None
        return run_prefix(pod, args.target_counter)
    identity = "ema-gram-prefix-" + args.mode + "-" + uuid.uuid4().hex[:12]
    tracker = WandbConfig(
        entity="marin-community",
        project="marin",
        name=identity,
        id=identity,
        group="ema-gram-20-prefix",
        mode="online",
        resume="never",
        background=False,
        save_code=False,
    ).init(None)
    print(f"W&B: https://wandb.ai/marin-community/marin/runs/{identity}", flush=True)
    try:
        tracker.log_hyperparameters(
            dict(
                mode=args.mode,
                source_sha256=source_hashes(),
                resolved_config=resolved,
                target_counter=args.target_counter,
            )
        )
        tracker.log({"logging_check/success": 1}, step=0)
        if args.mode == "cpu-check":
            cpu_check(tracker, args.output)
        else:
            args.output.write_text(
                json.dumps(dict(identity=identity, success=True, source_sha256=source_hashes()), indent=2) + "\n"
            )
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
