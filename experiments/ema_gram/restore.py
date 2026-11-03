# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit complete trainer restoration at the native LM training boundary, without training."""

import argparse
import hashlib
import json
import resource
import sys
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import draccus
import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts
from levanter.checkpoint import save_checkpoint
from levanter.checkpoint_manifest import read_manifest
from levanter.data.dataset import ListAsyncDataset
from levanter.models.qwen import Qwen3Config
from levanter.tensorstore_serialization import _create_ocdbt_spec, _flatten_serializable_leaves
from levanter.tracker import current_tracker
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import Trainer
from levanter.trainer_state import TrainerState
from marin.execution.lazy import materialized_config
from marin.training.training import run_levanter_train_lm
from rigging.filesystem.storage_path import StoragePath

from experiments.ema_gram.plan import Cell
from experiments.ema_gram.train import build_training_step


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def describe_arrays(tree):
    """Keep local shard indices explicit; hashes are not cross-sharding comparisons."""
    paths, arrays = _flatten_serializable_leaves(tree)
    result = []
    for path, array in zip(paths, arrays, strict=True):
        for shard in array.addressable_shards:
            host = np.asarray(shard.data)
            assert np.isfinite(host).all(), path
            result.append(dict(path=path, index=str(shard.index), shape=list(host.shape), sha256=digest(host)))
    return result


def audit_state(state, checkpoint, expected_step):
    """Compare every locally addressable restored shard against independent storage reads."""
    manifest = read_manifest(checkpoint)
    assert manifest is not None and manifest.array_driver == "zarr3" and manifest.kvstore_driver == "ocdbt"
    paths, arrays = _flatten_serializable_leaves(state.saveable_state)
    assert len(paths) == len(set(paths)) and set(paths) == manifest.array_paths
    entries = {entry.path: entry for entry in manifest.arrays}
    assert {path.split("/")[0] for path in paths} >= {"model", "opt_state", "step", "training_key"}
    assert int(state.step) == expected_step
    rows = []
    for path, array in zip(paths, arrays, strict=True):
        entry = entries[path]
        assert tuple(array.shape) == entry.shape and str(array.dtype) == entry.dtype, path
        store = ts.open(_create_ocdbt_spec(checkpoint, path), open=True).result()
        for shard in array.addressable_shards:
            actual = np.asarray(shard.data)
            saved = np.asarray(store[shard.index].read().result())
            assert actual.shape == saved.shape and actual.dtype == saved.dtype, path
            assert np.isfinite(actual).all(), path
            assert actual.tobytes() == saved.tobytes(), f"Restored bytes differ: {path} {shard.index}"
            rows.append(
                dict(
                    path=path,
                    index=str(shard.index),
                    device_id=shard.device.id,
                    bytes=actual.nbytes,
                    sha256=digest(actual),
                    storage_sha256=digest(saved),
                )
            )
            del actual, saved
    return dict(
        step=int(state.step),
        array_count=len(paths),
        local_shards=rows,
        all_manifest_paths_present=True,
        local_shards_match_storage_bytes=True,
        scope="Per-process addressable shards; require successful reports from every process",
    )


def audit_data(state, loader):
    step = int(state.step)
    assert loader._start_from_batch == step
    offset = loader.dl.scheduler.global_data_offset_by_step(step)
    first = next(loader)
    repeated = loader.dl.iter_from_step(step)
    try:
        again = next(repeated)
        a, b = describe_arrays(first), describe_arrays(again)
        assert a == b, "Restored data position does not replay the first batch"
    finally:
        repeated.close()
    return dict(
        start_step=step,
        global_example_offset=offset,
        first_batch=a,
        repeated_iterator_matches=True,
        scope="Matched configured stream at restored step; not a saved historical batch comparison",
    )


def memory():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(
        host_peak_rss_bytes=int(peak if sys.platform == "darwin" else peak * 1024),
        devices=[dict(id=d.id, stats=d.memory_stats()) for d in jax.local_devices()],
        scope="Restore/audit memory only; includes host copies for verification, no training step",
    )


def run_restore_probe(pod):
    """Run native train_lm initialization, replacing only entry into its training loop."""
    checkpoint = pod.train_config.trainer.initialize_from
    assert checkpoint and not pod.train_config.trainer.allow_partial_checkpoint
    identity = "ema-gram-restore-" + uuid.uuid4().hex[:12]
    tracker = replace(
        pod.train_config.trainer.tracker, id=identity, name=identity, mode="online", resume="never", background=False
    )
    pod = replace(
        pod, train_config=replace(pod.train_config, trainer=replace(pod.train_config.trainer, tracker=tracker))
    )
    metadata = json.loads((StoragePath(checkpoint) / "metadata.json").read_text())
    expected_step = metadata["step"]
    calls = []

    def inspect(trainer, state, loader):
        # Native initialization has already created the online tracker before PRNG/model kernels.
        assert jax.default_backend() == "tpu" and jax.process_count() == 4
        trainer.tracker.log({"restore_probe/entered": 1}, step=int(state.step))
        print(
            json.dumps(
                {"restore_phase": "native_state_loaded", "step": int(state.step), "process": jax.process_index()}
            ),
            flush=True,
        )
        report = dict(
            checkpoint=checkpoint,
            process_index=jax.process_index(),
            process_count=jax.process_count(),
            identity=identity,
            config=draccus.encode(pod),
            state=audit_state(state, checkpoint, expected_step),
        )
        try:
            report["data"] = audit_data(state, loader)
        finally:
            loader.close()
        report["memory"] = memory()
        report["training_steps_executed"] = 0
        report["full_training_memory_fit_verified"] = False
        destination = StoragePath(pod.output_path) / "restore-reports" / identity
        destination.mkdirs()
        (destination / f"process-{jax.process_index()}.json").write_text(json.dumps(report, indent=2))
        trainer.tracker.log(
            {"restore_probe/success": 1, "restore_probe/arrays": report["state"]["array_count"]}, step=int(state.step)
        )
        calls.append(report)
        print(
            json.dumps({"restore_report": str(destination), "process": jax.process_index(), "success": True}), flush=True
        )

    try:
        with patch.object(Trainer, "train", inspect):
            run_levanter_train_lm(pod)
        assert len(calls) == 1
    except BaseException:
        try:
            current_tracker().finish()
        except RuntimeError:
            pass
        raise


def cpu_check(tracker, output):
    """Exercise full native Trainer restore with a generated small checkpoint and data oracle."""
    assert jax.default_backend() == "cpu"
    model_config = Qwen3Config(
        hidden_dim=8, intermediate_dim=16, num_layers=2, num_heads=2, num_kv_heads=2, head_dim=4, max_seq_len=8
    )
    pod = materialized_config(build_training_step(Cell(0.25, 1e-4, 0.002), "restore-cpu-dev"), "/tmp/ema-restore")
    tx = pod.train_config.optimizer.build(7630)
    state = TrainerState.init(
        tx, model_config.build(hax.Axis("vocab", 16), key=jax.random.PRNGKey(1)), key=jax.random.PRNGKey(2)
    )
    grads = jax.tree.map(jnp.sin, state.trainable_model)
    update = eqx.filter_jit(lambda s: s.take_step(grads, key=jax.random.PRNGKey(3))[0])
    state = update(update(state))
    jax.block_until_ready(state)
    with tempfile.TemporaryDirectory(prefix="ema-full-restore-") as folder:
        source = str(Path(folder) / "step-2")
        save_checkpoint(state.saveable_state, 2, source)
        cfg = replace(
            pod.train_config.trainer,
            id="restore-cpu",
            initialize_from=source,
            load_checkpoint=False,
            allow_partial_checkpoint=False,
            mesh=replace(pod.train_config.trainer.mesh, axes={"data": 1, "model": 1, "replica": 1}),
            train_batch_size=2,
            require_accelerator=False,
        )
        with current_tracker(tracker), Trainer(cfg, tx, lambda *a, **kw: None, add_default_hooks=False) as trainer:
            loaded = trainer.initial_state(
                jax.random.PRNGKey(99),
                model_init=lambda: model_config.build(hax.Axis("vocab", 16), key=jax.random.PRNGKey(77)),
            )
            report = dict(state=audit_state(loaded, source, 2))
            assert describe_arrays(loaded.saveable_state) == describe_arrays(state.saveable_state)
            dataset = ListAsyncDataset([np.array([i], dtype=np.int32) for i in range(16)])
            loader = trainer.data_loader(dataset).iter_from_step(2)
            try:
                report["data"] = audit_data(loaded, loader)
            finally:
                loader.close()
            oracle = np.array([[4], [5]], dtype=np.int32)
            assert report["data"]["global_example_offset"] == 4
            assert report["data"]["first_batch"][0]["sha256"] == digest(oracle)
            before = describe_arrays(loaded.saveable_state)
            changed = update(loaded)
            assert int(changed.step) == 3 and describe_arrays(loaded.saveable_state) == before
            # Auditing a mutated restored state against the source must fail.
            try:
                audit_state(replace(loaded, training_key=jax.random.PRNGKey(123)), source, 2)
            except AssertionError:
                report["mutated_prng_rejected"] = True
            else:
                raise AssertionError("Mutated PRNG was accepted")
            report["unchanged_input_after_synthetic_update"] = True
            report["synthetic_update_all_finite"] = bool(describe_arrays(changed.saveable_state))
        report["scope"] = "Generated tiny CPU trainer checkpoint; original checkpoint and TPU untested"
        report["memory"] = memory()
        output.write_text(json.dumps(report, indent=2) + "\n")
        tracker.log({"restore_cpu/success": 1}, step=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("logging-check", "cpu-check", "prepare", "worker"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--source-sha256")
    parser.add_argument("--version", default="2026.11.03.23")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode in ("logging-check", "cpu-check"):
        identity = "ema-gram-restore-" + args.mode + "-" + uuid.uuid4().hex[:12]
        tracker = WandbConfig(
            entity="marin-community",
            project="marin",
            name=identity,
            id=identity,
            group="ema-gram-20-restore",
            mode="online",
            resume="never",
            background=False,
            save_code=False,
        ).init(None)
        print(f"W&B: https://wandb.ai/marin-community/marin/runs/{identity}", flush=True)
        try:
            tracker.log_hyperparameters(
                dict(mode=args.mode, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            )
            tracker.log({"logging_check/success": 1}, step=0)
            if args.mode == "cpu-check":
                cpu_check(tracker, args.output)
            else:
                args.output.write_text(json.dumps(dict(mode=args.mode, success=True, identity=identity)) + "\n")
        finally:
            tracker.finish()
    else:
        assert args.checkpoint
        step = build_training_step(
            Cell(0.25, 1e-4, 0.002), args.version, tpu_type="v4-32", initialize_from=args.checkpoint
        )
        pod = materialized_config(step, "gs://marin-us-central2")
        # Audit host copies need headroom beyond the prior 128 GiB failed synthetic job.
        pod = replace(pod, resources=replace(pod.resources, ram="256g"))
        args.output.write_text(json.dumps(draccus.encode(pod), indent=2) + "\n")
        if args.mode == "worker":
            assert args.source_sha256 == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            evidence = json.loads(Path("logbooks/20-ema-gram-polar/loss-pair-checkpoint-metadata.json").read_text())
            assert args.checkpoint == evidence["checkpoint"]
            for name, expected in evidence["files"].items():
                raw = (StoragePath(args.checkpoint) / name).read_bytes()
                assert len(raw) == expected["bytes"] and hashlib.sha256(raw).hexdigest() == expected["sha256"]
            run_restore_probe(pod)


if __name__ == "__main__":
    main()
