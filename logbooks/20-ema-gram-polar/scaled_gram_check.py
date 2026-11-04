# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test whether more matrix-parallel shards accelerate full grouped Gram solves."""

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
    grouped_gram_powers,
    init_gram_state,
    matrix_parallel_eigh,
    relative_gram_power,
    scale_by_ema_gram,
)
from levanter.tracker.wandb import WandbTracker
from optimizer_cost import NAMES, ROW_SHARDED, checkpoint_states, put_global, refresh
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret
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
    if layout == "matrix":
        fn = partial(matrix_parallel_eigh, mesh=mesh, axis_name=tuple(mesh.axis_names))
        return tuple(gram_direction(s, alpha=0.25, ridge=1e-4, polar_steps=5, eigh_fn=fn) for s in states)
    powers = grouped_gram_powers(
        tuple(g for state in states for g in (state.left, state.right)),
        alpha=0.25,
        ridge=1e-4,
        mesh=mesh,
        axis_name=tuple(mesh.axis_names),
    )
    return tuple(
        gram_direction(state, alpha=0.25, ridge=1e-4, polar_steps=5, precomputed_powers=(left, right))
        for state, left, right in zip(states, powers[::2], powers[1::2], strict=True)
    )


def native_update(gradients, states, *, layout, alpha=0.25):
    optimizer = scale_by_ema_gram(alpha=alpha, ridge=1e-4, eigh_layout=layout)
    directions, updated = optimizer.update(gradients, EmaGramState(states))
    return directions, updated.matrices


def cpu_power_cases(mesh):
    """Mixed shapes, zero/rank-deficient banks and uneven batches versus NumPy."""
    rng = np.random.default_rng(41)
    grams = []
    for batch, size in [(3, 8), (2, 4), (2, 8)]:
        matrix = rng.standard_normal((batch, size, size), dtype=np.float32)
        grams.append(matrix @ matrix.swapaxes(-1, -2))
    grams[0][0] = 0
    grams[0][1] = np.ones((8, 8), dtype=np.float32)
    before = [g.tobytes() for g in grams]
    fn = jax.jit(partial(grouped_gram_powers, alpha=0.25, ridge=1e-4, mesh=mesh, axis_name=tuple(mesh.axis_names)))
    inputs = tuple(put_global(g, NamedSharding(mesh, P())) for g in grams)
    outputs = jax.block_until_ready(fn(inputs))
    diagnostics = []
    for bank, output in zip(grams, outputs, strict=True):
        assert np.isfinite(np.asarray(output)).all()
        for gram, actual in zip(bank, np.asarray(output), strict=True):
            normalized = gram.astype(np.float64)
            normalized = (normalized + normalized.T) * 0.5
            mean = np.trace(normalized) / len(normalized)
            if mean == 0:
                np.testing.assert_array_equal(actual, np.eye(len(gram)))
                continue
            if np.linalg.matrix_rank(gram) < len(gram):
                continue  # Rank-deficient case checks finiteness; full-rank banks use the independent oracle.
            normalized /= mean
            values, vectors = np.linalg.eigh(normalized)
            expected = (vectors * ((np.maximum(values, 0) + 1e-4) ** -0.25)[None, :]) @ vectors.T
            diagnostics.append(
                dict(
                    max_absolute_error=float(np.max(np.abs(actual - expected))),
                    passed=bool(np.allclose(actual, expected, rtol=1e-5, atol=1e-5)),
                )
            )
    baseline_fn = jax.jit(
        partial(
            relative_gram_power,
            alpha=0.25,
            ridge=1e-4,
            eigh_fn=partial(matrix_parallel_eigh, mesh=mesh, axis_name=tuple(mesh.axis_names)),
        )
    )
    for bank, actual in zip(inputs, outputs, strict=True):
        baseline = np.asarray(jax.block_until_ready(baseline_fn(bank)))
        np.testing.assert_allclose(np.asarray(actual), baseline, rtol=1e-5, atol=1e-5)
    diagonal = tuple(
        np.stack([np.diag(np.arange(1, size + 1, dtype=np.float32) + i) for i in range(batch)])
        for batch, size in [(3, 8), (2, 4), (2, 8)]
    )
    known = jax.block_until_ready(fn(tuple(put_global(g, NamedSharding(mesh, P())) for g in diagonal)))
    for bank, actual in zip(diagonal, known, strict=True):
        for gram, output in zip(bank, np.asarray(actual), strict=True):
            values = np.diag(gram).astype(np.float64)
            expected = np.diag((values / values.mean() + 1e-4) ** -0.25)
            np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)
    assert [g.tobytes() for g in grams] == before
    return dict(
        zero_exact_identity=True,
        rank_deficient_finite=True,
        independent_numpy_diagonal_reference=True,
        random_full_rank_reference_diagnostics=diagnostics,
        grouped_matches_independent_bank_baseline=True,
        input_unchanged=True,
        mixed_bank_shapes=[[3, 8, 8], [2, 4, 4], [2, 8, 8]],
        grouped_counts={"8": 5, "4": 2},
        nondivisible_group_padding=True,
    )


def execute(args, tracker, persist):
    cpu = args.mode == "cpu-check"
    assert jax.default_backend() == ("cpu" if cpu else "tpu")
    devices = args.mesh_data * args.mesh_model
    assert jax.device_count() == devices
    assert jax.process_count() == args.processes
    if not cpu:
        assert devices in (32, 64) and all("v4" in d.device_kind.lower() for d in jax.devices())
    mesh = Mesh(np.array(jax.devices()).reshape(args.mesh_data, args.mesh_model), ("data", "model"))
    axes = tuple(mesh.axis_names)
    replicated = NamedSharding(mesh, P())
    gradients_sharding = tuple(
        NamedSharding(mesh, P(None, axes, None) if n in ROW_SHARDED else P(None, None, axes)) for n in NAMES
    )
    states_sharding = tuple(GramState(replicated, replicated, replicated) for _ in NAMES)
    layers, width = (3, devices) if cpu else (11, 1024)
    assert width % devices == 0
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
            device_count=jax.device_count(),
            device_kinds=sorted({d.device_kind for d in jax.devices()}),
            mesh_shape=dict(mesh.shape),
            source_assignment_counts={
                str(size): dict(
                    real_matrices=banks * layers,
                    independent_padded_matrices=banks * math.ceil(layers / devices) * devices,
                    grouped_padded_matrices=math.ceil(banks * layers / devices) * devices,
                    independent_max_matrices_per_shard=banks * math.ceil(layers / devices),
                    grouped_max_matrices_per_shard=math.ceil(banks * layers / devices),
                )
                for size, banks in [(width, 11), (width * 4, 3)]
            },
            source_files=source_files(),
            source_commit=os.environ.get("GIT_COMMIT"),
            checkpoint=evidence,
            shapes=shapes,
            gradient_seed=0,
            gradient_scale=scale,
            cases={},
            completed=False,
            reference="matrix",
            candidate="matrix_grouped",
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
            for layout in ["matrix", "matrix_grouped"]:
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
            for layout in ["matrix", "matrix_grouped"]:
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
        tracker.log({"scaled_gram/success": 1}, step=metric_step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["logging-check", "cpu-check", "worker"], required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--mesh-data", type=int, required=True)
    parser.add_argument("--mesh-model", type=int, required=True)
    parser.add_argument("--processes", type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.source_manifest.read_text())
    assert manifest["files"] == source_files() and args.samples == 3
    assert args.mesh_data > 0 and args.mesh_model > 0 and args.processes > 0
    if args.mode == "cpu-check":
        assert args.mesh_data * args.mesh_model in (8, 16) and args.processes == 1
    elif args.mode == "worker":
        assert args.mesh_data * args.mesh_model in (32, 64)
        assert args.processes * 4 == args.mesh_data * args.mesh_model
    metadata = ROOT / "logbooks/20-ema-gram-polar/loss-pair-checkpoint-metadata.json"
    assert hashlib.sha256(metadata.read_bytes()).hexdigest() == manifest["checkpoint_metadata_sha256"]
    if args.mode == "worker":
        assert args.checkpoint
        initialize_jax()
    identity = "ema-gram-scaled-" + args.mode + "-" + uuid.uuid4().hex[:12]
    destination = StoragePath(args.output_prefix) / identity
    destination.mkdirs()
    # Each fresh process owns an online run, using Levanter's tracker adapter.
    run = wandb.init(
        entity="marin-community",
        project="marin",
        id=identity,
        name=identity,
        group="ema-gram-20-scaled-gram",
        mode="online",
        resume="never",
        settings={"api_key": require_secret("WANDB_API_KEY")},
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
        mesh_shape={"data": args.mesh_data, "model": args.mesh_model},
        expected_processes=args.processes,
        reference="matrix",
        candidate="matrix_grouped",
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
