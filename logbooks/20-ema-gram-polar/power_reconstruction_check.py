# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare gathered eigenpairs with shard-local power reconstruction, without training."""

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import tempfile
import time
import uuid
from functools import partial
from pathlib import Path

import jax
import numpy as np
import wandb
from iris.jax.init import initialize_jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import save_checkpoint
from levanter.optim.ema_gram import (
    EmaGramState,
    GramState,
    gram_direction,
    init_gram_state,
    matrix_parallel_eigh,
    matrix_parallel_power,
    relative_gram_power,
    scale_by_ema_gram,
)
from levanter.tracker.wandb import WandbTracker
from optimizer_cost import NAMES, ROW_SHARDED, checkpoint_states, put_global, refresh
from rigging.filesystem.storage_path import StoragePath
from saved_state_check import compare, fingerprints, snapshot

ROOT = Path(__file__).resolve().parents[2]


def source_files():
    paths = [
        Path(__file__).relative_to(ROOT),
        Path("logbooks/20-ema-gram-polar/saved_state_check.py"),
        Path("logbooks/20-ema-gram-polar/optimizer_cost.py"),
        Path("lib/levanter/src/levanter/optim/ema_gram.py"),
    ]
    return {
        str(p): dict(bytes=len(raw := (ROOT / p).read_bytes()), sha256=hashlib.sha256(raw).hexdigest()) for p in paths
    }


def unchanged_directions(states, *, mesh, layout):
    kwargs = (
        dict(eigh_fn=partial(matrix_parallel_eigh, mesh=mesh, axis_name=tuple(mesh.axis_names)))
        if layout == "matrix"
        else dict(power_fn=partial(matrix_parallel_power, mesh=mesh, axis_name=tuple(mesh.axis_names)))
    )
    return tuple(gram_direction(s, alpha=0.25, ridge=1e-4, polar_steps=5, **kwargs) for s in states)


def native_update(gradients, states, *, layout, alpha=0.25):
    optimizer = scale_by_ema_gram(alpha=alpha, ridge=1e-4, eigh_layout=layout)
    directions, updated = optimizer.update(gradients, EmaGramState(states))
    return directions, updated.matrices


def cpu_power_cases(mesh):
    """Independent CPU spectral reference, including zero and rank-deficient inputs."""
    rng = np.random.default_rng(41)
    matrix = rng.standard_normal((3, 8, 8), dtype=np.float32)
    gram = matrix @ matrix.swapaxes(-1, -2)
    gram[0] = 0
    gram[1] = np.ones((8, 8), dtype=np.float32)
    before = gram.tobytes()
    fn = jax.jit(
        partial(
            relative_gram_power,
            alpha=0.25,
            ridge=1e-4,
            power_fn=partial(matrix_parallel_power, mesh=mesh, axis_name=tuple(mesh.axis_names)),
        )
    )
    output = np.asarray(jax.block_until_ready(fn(put_global(gram, NamedSharding(mesh, P())))))
    assert np.isfinite(output).all() and gram.tobytes() == before
    np.testing.assert_array_equal(output[0], np.eye(8, dtype=np.float32))
    normalized = gram[2].astype(np.float64)
    normalized = (normalized + normalized.T) * 0.5
    normalized /= np.trace(normalized) / 8
    values, vectors = np.linalg.eigh(normalized)
    reference = (vectors * ((np.maximum(values, 0) + 1e-4) ** -0.25)[None, :]) @ vectors.T
    np.testing.assert_allclose(output[2], reference, rtol=1e-5, atol=1e-5)
    return dict(
        zero_exact_identity=True,
        rank_deficient_finite=True,
        independent_numpy_full_rank_reference=True,
        input_unchanged=True,
        nondivisible_batch=3,
    )


def execute(args, tracker, persist):
    cpu = args.mode == "cpu-check"
    assert jax.default_backend() == ("cpu" if cpu else "tpu")
    assert jax.device_count() == (4 if cpu else 16)
    if not cpu:
        assert jax.process_count() == 4 and all("v4" in d.device_kind.lower() for d in jax.devices())
    side = 2 if cpu else 4
    mesh = Mesh(np.array(jax.devices()).reshape(side, side), ("data", "model"))
    axes = tuple(mesh.axis_names)
    replicated = NamedSharding(mesh, P())
    gradients_sharding = tuple(
        NamedSharding(mesh, P(None, axes, None) if n in ROW_SHARDED else P(None, None, axes)) for n in NAMES
    )
    states_sharding = tuple(GramState(replicated, replicated, replicated) for _ in NAMES)
    layers, width = (3, 8) if cpu else (11, 1024)
    shapes = [(layers, width, width)] * 4 + [(layers, width * 4, width)] * 2 + [(layers, width, width * 4)]
    rng = np.random.default_rng(0)
    scale = 0.1 / math.sqrt(sum(math.prod(s) for s in shapes))
    gradients = tuple(
        put_global(rng.standard_normal(s, dtype=np.float32) * scale, sh)
        for s, sh in zip(shapes, gradients_sharding, strict=True)
    )
    with tempfile.TemporaryDirectory(prefix="ema-power-reconstruction-") as temporary, jax.set_mesh(mesh):
        if cpu:
            states = jax.block_until_ready(
                jax.jit(refresh, out_shardings=states_sharding)(gradients, tuple(init_gram_state(g) for g in gradients))
            )
            expected = fingerprints(states)
            checkpoint = str(Path(temporary) / "step-0")
            save_checkpoint({"saved": EmaGramState(states)}, 0, checkpoint)
            del states
        else:
            checkpoint = args.checkpoint
            identities = json.loads((Path(__file__).parent / "loss-pair-checkpoint-metadata.json").read_text())
            assert checkpoint == identities["checkpoint"]
            for name, identity in identities["files"].items():
                raw = (StoragePath(checkpoint) / name).read_bytes()
                assert len(raw) == identity["bytes"] and hashlib.sha256(raw).hexdigest() == identity["sha256"]
        states, evidence = checkpoint_states(checkpoint, shapes, mesh)
        before, gradient_before = fingerprints(states), fingerprints(gradients)
        if cpu:
            assert before == expected
        report = dict(
            mode=args.mode,
            process_index=jax.process_index(),
            process_count=jax.process_count(),
            source_files=source_files(),
            source_commit=os.environ.get("GIT_COMMIT"),
            checkpoint=evidence,
            shapes=shapes,
            gradient_seed=0,
            gradient_scale=scale,
            cases={},
            completed=False,
            reference="matrix",
            candidate="matrix_power",
            samples=args.samples,
            training_steps=0,
            full_trainer_restore=False,
            full_training_memory_fit=False,
            equivalence_is_training_veto=False,
            scope="Optimizer-only; timings exclude training and are not MFU",
        )
        persist(report)
        if cpu:
            report["cpu_power_cases"] = cpu_power_cases(mesh)
        metric_step = 1
        for case, full in [("unchanged_saved_state_direction", False), ("fixed_synthetic_gradient_update", True)]:
            inputs = (gradients, states) if full else (states,)
            input_shardings = (gradients_sharding, states_sharding) if full else (states_sharding,)
            output_shardings = (gradients_sharding, states_sharding) if full else gradients_sharding
            reference = None
            case_report = {}
            for layout in ["matrix", "matrix_power"]:
                function = (
                    partial(native_update, layout=layout)
                    if full
                    else partial(unchanged_directions, mesh=mesh, layout=layout)
                )
                fn = jax.jit(function, in_shardings=input_shardings, out_shardings=output_shardings)
                tracker.log({f"{case}/{layout}/started": 1}, step=metric_step)
                metric_step += 1
                begin = time.perf_counter()
                output = jax.block_until_ready(fn(*inputs))
                compile_and_first = time.perf_counter() - begin
                arrays = snapshot(output)
                assert len({leaf for leaf, _ in arrays}) == (28 if full else 7)
                del output
                samples, replay_equal = [], []
                for _ in range(args.samples):
                    begin = time.perf_counter()
                    output = jax.block_until_ready(fn(*inputs))
                    samples.append(time.perf_counter() - begin)
                    # Outside the measured interval; finite/replica checks apply to every sample.
                    repeated = snapshot(output)
                    replay_equal.append(
                        {k: v.tobytes() for k, v in repeated.items()} == {k: v.tobytes() for k, v in arrays.items()}
                    )
                    del output, repeated
                fn.clear_cache()
                del fn
                gc.collect()
                assert fingerprints(states) == before and fingerprints(gradients) == gradient_before
                case_report[layout] = dict(
                    finite=True,
                    input_immutable=True,
                    bitwise_replays=sum(replay_equal),
                    replay_comparisons=replay_equal,
                    compile_and_first_seconds=compile_and_first,
                    warmed_sample_seconds=samples,
                )
                if layout == "matrix":
                    reference = arrays
                else:
                    case_report["comparison"] = compare(arrays, reference)
                    del arrays
                tracker.log(
                    {f"{case}/{layout}/finite": 1, f"{case}/{layout}/mean_optimizer_seconds": float(np.mean(samples))},
                    step=metric_step,
                )
                metric_step += 1
                report["cases"][case] = case_report
                persist(report)
            del reference
            gc.collect()
        if cpu:
            alpha_zero = []
            for layout in ["matrix", "matrix_power"]:
                output = jax.block_until_ready(
                    jax.jit(partial(native_update, layout=layout, alpha=0.0))(gradients, states)
                )
                alpha_zero.append(fingerprints(output))
                del output
            assert alpha_zero[0] == alpha_zero[1]
            report["cpu_alpha_zero_exact"] = True
        report["completed"] = True
        report["peak_rss_native_units"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["rss_units"] = "bytes on Darwin; KiB on Linux; optimizer harness only"
        persist(report)
        tracker.log({"power_reconstruction/success": 1}, step=metric_step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["logging-check", "cpu-check", "worker"], required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads(args.source_manifest.read_text())
    assert manifest["files"] == source_files() and args.samples > 0
    metadata = ROOT / "logbooks/20-ema-gram-polar/loss-pair-checkpoint-metadata.json"
    assert hashlib.sha256(metadata.read_bytes()).hexdigest() == manifest["checkpoint_metadata_sha256"]
    if args.mode == "worker":
        assert args.checkpoint
        initialize_jax()
    identity = "ema-gram-power-" + args.mode + "-" + uuid.uuid4().hex[:12]
    destination = StoragePath(args.output_prefix) / identity
    destination.mkdirs()
    # Each fresh process owns an online run, using Levanter's tracker adapter.
    run = wandb.init(
        entity="marin-community",
        project="marin",
        id=identity,
        name=identity,
        group="ema-gram-20-power-reconstruction",
        mode="online",
        resume="never",
    )
    assert run.settings.mode == "online"
    tracker = WandbTracker(run, replicate_path=str(destination))
    config = dict(
        mode=args.mode,
        identity=identity,
        source_files=source_files(),
        source_commit=os.environ.get("GIT_COMMIT"),
        process_index=jax.process_index(),
        samples=args.samples,
        reference="matrix",
        candidate="matrix_power",
    )
    tracker.log_hyperparameters(config)
    tracker.log({"logging_check/success": 1}, step=0)
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    print(f"W&B: {run.url}", flush=True)
    try:
        if args.mode != "logging-check":
            execute(args, tracker, lambda report: (destination / "report.json").write_text(json.dumps(report, indent=2)))
            print(json.dumps(dict(completed=True, report=str(destination / "report.json"))), flush=True)
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        tracker.finish()


if __name__ == "__main__":
    main()
