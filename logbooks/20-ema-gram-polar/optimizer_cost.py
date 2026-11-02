# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure EMA-Gram kernels on synthetic inputs with the pilot's matrix geometry.

This is a non-training diagnostic. An optional checkpoint supplies saved Gram
and momentum arrays; gradients remain synthetic. Neither mode measures
language-model quality or full-training MFU.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import time
import uuid
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from iris.jax.init import initialize_jax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import load_checkpoint
from levanter.checkpoint_manifest import read_manifest
from levanter.optim.ema_gram import (
    GramState,
    gram_direction,
    init_gram_state,
    matrix_parallel_eigh,
    relative_gram_power,
    update_gram_state,
)
from levanter.tracker.wandb import WandbConfig
from marin.profiling.xplane import _xspace_message_class
from rigging.filesystem.storage_path import StoragePath

NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ROW_SHARDED = {"o_proj", "down_proj"}


def put_global(array, sharding):
    return jax.make_array_from_callback(array.shape, sharding, lambda index: array[index])


def refresh(gradients, states):
    return tuple(update_gram_state(g, s, 0.95, 0.99) for g, s in zip(gradients, states, strict=True))


def powers(states, *, eigh_fn=jax.lax.linalg.eigh, eigh_implementation=None):
    power = partial(
        relative_gram_power, alpha=0.25, ridge=1e-4, eigh_fn=eigh_fn, eigh_implementation=eigh_implementation
    )
    return tuple((power(s.left), power(s.right)) for s in states)


def directions(states, alpha, *, eigh_fn=jax.lax.linalg.eigh, eigh_implementation=None):
    return tuple(
        gram_direction(
            s, alpha=alpha, ridge=1e-4, polar_steps=5, eigh_fn=eigh_fn, eigh_implementation=eigh_implementation
        )
        for s in states
    )


def full_update(gradients, states, *, eigh_fn=jax.lax.linalg.eigh, eigh_implementation=None):
    updated = refresh(gradients, states)
    return directions(updated, 0.25, eigh_fn=eigh_fn, eigh_implementation=eigh_implementation), updated


def checkpoint_states(checkpoint, shapes, mesh):
    """Read only saved Gram/momentum arrays; this is not a full trainer restore."""
    manifest = read_manifest(checkpoint)
    if manifest is None:
        raise ValueError("Checkpoint diagnostic requires an array manifest")
    fields = GramState._fields
    entries = [a for a in manifest.arrays if "/matrices/" in a.path and a.path.rsplit("/", 1)[-1] in fields]
    prefixes = {a.path.split("/matrices/", 1)[0] for a in entries}
    if len(prefixes) != 1:
        raise ValueError(f"Expected one EMA-Gram state bank, found {prefixes}")
    prefix = prefixes.pop() + "/matrices"
    by_path = {a.path: a for a in entries}
    expected_paths = [f"{prefix}/{i}/{field}" for i in range(len(shapes)) for field in fields]
    if set(by_path) != set(expected_paths):
        raise ValueError("Checkpoint matrix count or fields differ from diagnostic geometry")
    replicated = NamedSharding(mesh, P())
    exemplar = {}
    for index, shape in enumerate(shapes):
        batch, rows, cols = shape[:-2], shape[-2], shape[-1]
        expected_shapes = (tuple(shape), (*batch, rows, rows), (*batch, cols, cols))
        for field, expected_shape in zip(fields, expected_shapes, strict=True):
            path = f"{prefix}/{index}/{field}"
            entry = by_path[path]
            if entry.shape != expected_shape or entry.dtype != "float32":
                raise ValueError(f"Unexpected checkpoint array: {entry}")
            exemplar[path] = jax.ShapeDtypeStruct(entry.shape, np.float32, sharding=replicated)
    loaded = load_checkpoint(exemplar, checkpoint, mesh=mesh, axis_mapping={}, allow_partial=False)
    states = tuple(GramState(*(loaded[f"{prefix}/{i}/{field}"] for field in fields)) for i in range(len(shapes)))
    metadata = json.loads((StoragePath(checkpoint) / "metadata.json").read_text())
    evidence = {
        "checkpoint_uri": checkpoint,
        "metadata": metadata,
        "loaded_paths": expected_paths,
        "loaded_bytes": sum(math.prod(a.shape) * np.dtype(a.dtype).itemsize for a in entries),
        "manifest_sha256": hashlib.sha256((StoragePath(checkpoint) / "manifest.json").read_bytes()).hexdigest(),
        "full_trainer_restore": False,
        "scope": "Saved FP32 Gram and momentum arrays only; model, AdamW, step and data state are not restored",
    }
    return states, evidence


def compare_trees(actual, expected):
    """Record every leaf at the fixed numerical gate without suppressing failures."""
    comparisons = []
    for index, (a, b) in enumerate(zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True)):
        comparisons.append(
            {
                "leaf": index,
                "max_absolute_error": float(jnp.max(jnp.abs(a - b))),
                "relative_frobenius_error": float(jnp.linalg.norm(a - b) / jnp.maximum(jnp.linalg.norm(b), 1e-30)),
                "passed": bool(jnp.all(jnp.abs(a - b) <= 1e-5 + 1e-5 * jnp.abs(b))),
            }
        )
    return {"rtol": 1e-5, "atol": 1e-5, "passed": all(row["passed"] for row in comparisons), "leaves": comparisons}


def finite_tree(tree):
    return all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(tree))


def measure(fn, inputs, samples):
    start = time.perf_counter()
    output = jax.block_until_ready(fn(*inputs))
    warmup = time.perf_counter() - start
    assert finite_tree(output), "Nonfinite warm-up output"
    del output
    durations = []
    for _ in range(samples):
        start = time.perf_counter()
        output = jax.block_until_ready(fn(*inputs))
        durations.append(time.perf_counter() - start)
        assert finite_tree(output), "Nonfinite timed output"
        del output
    return {
        "warmup_compile_and_execution_seconds": warmup,
        "sample_seconds": durations,
        "median_seconds": statistics.median(durations),
        "finite_outputs": True,
    }


def profile_once(fn, inputs, directory):
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 1
    options.python_tracer_level = 0
    options.enable_hlo_proto = False
    with jax.profiler.trace(str(directory), create_perfetto_trace=False, profiler_options=options):
        with jax.profiler.StepTraceAnnotation("ema_gram_optimizer_diagnostic", step_num=0):
            output = jax.block_until_ready(fn(*inputs))
    assert finite_tree(output)
    del output
    records = []
    for path in directory.rglob("*.xplane.pb"):
        content = path.read_bytes()
        space = _xspace_message_class()()
        space.ParseFromString(content)
        records.append(
            {
                "relative_path": str(path.relative_to(directory)),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "planes": [{"name": p.name, "events": sum(len(l.events) for l in p.lines)} for p in space.planes],
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tpu-type", choices=("v4-8", "v4-32", "v5litepod-8"), default="v4-32")
    parser.add_argument("--eigh-layout", choices=("replicated", "matrix"), default="replicated")
    parser.add_argument("--eigh-implementation", choices=("default", "jacobi"), default="default")
    parser.add_argument("--checkpoint", help="Read saved Gram/momentum only; no full restore or training")
    parser.add_argument("--smoke", action="store_true", help="Small CPU geometry; never performance evidence")
    parser.add_argument("--logging-check", action="store_true", help="Online tracker only; no kernel execution")
    args = parser.parse_args()
    assert args.samples > 0
    if not args.smoke:
        initialize_jax()
    rank = jax.process_index()
    identity_bytes = np.frombuffer(uuid.uuid4().bytes, dtype=np.uint8)
    identity_bytes = multihost_utils.broadcast_one_to_all(identity_bytes)
    attempt = bytes(identity_bytes).hex()[:12]
    mode = "logging-check" if args.logging_check else "cpu-smoke" if args.smoke else "tpu-cost"
    identity = f"ema-gram-20-{mode}-{attempt}"
    local = Path("/tmp") / identity
    local.mkdir(parents=True)
    run_destination = StoragePath(args.output_prefix) / identity
    destination = run_destination / f"rank-{rank}"
    destination.mkdirs()
    layers, width = (3, 8) if args.smoke else (11, 1024)
    shapes = [(layers, width, width)] * 4 + [(layers, width * 4, width)] * 2 + [(layers, width, width * 4)]
    config = {
        "mode": mode,
        "tpu_type": args.tpu_type,
        "eigh_layout": args.eigh_layout,
        "eigh_implementation": args.eigh_implementation,
        "process_count": jax.process_count(),
        "process_index": rank,
        "source_commit": os.environ.get("GIT_COMMIT"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": args.seed,
        "samples": args.samples,
        "matrix_names": NAMES,
        "matrix_shapes": shapes,
        "alpha": 0.25,
        "ridge": 1e-4,
        "beta1": 0.95,
        "beta2": 0.99,
        "polar_steps": 5,
        "state_sharding": "replicated",
        "gradient_sharding": "embed axis on data; o/down rows, others columns",
        "input_state": (
            "saved checkpoint Gram/momentum; synthetic gradients; same inputs for every timing"
            if args.checkpoint
            else "one synthetic gradient update from zero; same inputs for every timing"
        ),
        "checkpoint_uri": args.checkpoint,
        "limitations": (
            "No model/data/AdamW/weight decay. Synthetic gradients; "
            "isolated kernels do not establish full-training time fractions."
        ),
        "profile_invocations": 1,
        "output_prefix": str(destination),
    }
    tracker = WandbConfig(
        entity="marin-community",
        project="marin",
        name=identity,
        id=identity,
        group="ema-gram-20-cost-diagnostic",
        mode="online",
        resume="never",
        background=False,
        save_code=False,
        replicate_path=str(run_destination),
    ).init(None)
    print(f"W&B: https://wandb.ai/marin-community/marin/runs/{identity}", flush=True)
    tracker.log_hyperparameters(config)
    tracker.log({"logging_check/success": 1}, step=0)
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    if args.logging_check:
        tracker.finish()
        return
    devices = jax.devices()
    if not args.smoke:
        expected_devices, expected_processes = {"v4-8": (4, 1), "v4-32": (16, 4), "v5litepod-8": (8, 1)}[args.tpu_type]
        assert (len(devices), jax.process_count()) == (expected_devices, expected_processes), devices
        expected_kind = "v4" if args.tpu_type.startswith("v4-") else "v5"
        assert all(d.platform == "tpu" and expected_kind in d.device_kind.lower() for d in devices), devices
        if args.tpu_type == "v5litepod-8":
            assert all("lite" in d.device_kind.lower() or "v5e" in d.device_kind.lower() for d in devices), devices
    mesh = Mesh(np.array(devices), ("data",))
    replicated = NamedSharding(mesh, P())
    solver = None if args.eigh_implementation == "default" else jax.lax.linalg.EighImplementation.JACOBI
    eig_fn = (
        jax.lax.linalg.eigh
        if args.eigh_layout == "replicated"
        else partial(matrix_parallel_eigh, mesh=mesh, axis_name="data")
    )
    solver_kwargs = dict(eigh_fn=eig_fn, eigh_implementation=solver)
    gradient_shardings = tuple(
        NamedSharding(mesh, P(None, "data", None) if n in ROW_SHARDED else P(None, None, "data")) for n in NAMES
    )
    state_shardings = tuple(GramState(replicated, replicated, replicated) for _ in NAMES)
    config["devices"] = [{"id": d.id, "platform": d.platform, "kind": d.device_kind} for d in devices]
    tracker.log_hyperparameters(config)
    rng = np.random.default_rng(args.seed)
    scale = 0.1 / math.sqrt(sum(math.prod(shape) for shape in shapes))
    gradients = tuple(
        put_global(rng.standard_normal(shape, dtype=np.float32) * scale, sharding)
        for shape, sharding in zip(shapes, gradient_shardings, strict=True)
    )
    refresh_fn = jax.jit(refresh, in_shardings=(gradient_shardings, state_shardings), out_shardings=state_shardings)
    if args.checkpoint:
        states, config["checkpoint_input"] = checkpoint_states(args.checkpoint, shapes, mesh)
        assert finite_tree(states), "Checkpoint state contains nonfinite values"
        tracker.log_hyperparameters(config)
        (destination / "config.json").write_text(json.dumps(config, indent=2))
    else:
        prior = tuple(
            put_global(rng.standard_normal(shape, dtype=np.float32) * scale, sharding)
            for shape, sharding in zip(shapes, gradient_shardings, strict=True)
        )
        initialize = jax.jit(lambda gs: tuple(init_gram_state(g) for g in gs), out_shardings=state_shardings)
        states = jax.block_until_ready(refresh_fn(prior, initialize(prior)))
        del prior
    updated = jax.block_until_ready(refresh_fn(gradients, states))
    full_fn = jax.jit(
        partial(full_update, **solver_kwargs),
        in_shardings=(gradient_shardings, state_shardings),
        out_shardings=(gradient_shardings, state_shardings),
    )
    cases = [
        ("gram_refresh", refresh_fn, (gradients, states)),
        (
            "spectral_powers",
            jax.jit(partial(powers, **solver_kwargs), in_shardings=(state_shardings,), out_shardings=replicated),
            (updated,),
        ),
        (
            "alpha0_direction",
            jax.jit(
                partial(directions, alpha=0, **solver_kwargs),
                in_shardings=(state_shardings,),
                out_shardings=gradient_shardings,
            ),
            (updated,),
        ),
        ("quarter_update_and_direction", full_fn, (gradients, states)),
    ]
    if args.checkpoint:
        checkpoint_direction_fn = jax.jit(
            partial(directions, alpha=0.25, **solver_kwargs),
            in_shardings=(state_shardings,),
            out_shardings=gradient_shardings,
        )
        cases.append(("saved_checkpoint_direction", checkpoint_direction_fn, (states,)))
    report = {"config": config, "measurements": {}}
    for step, (name, fn, inputs) in enumerate(cases, start=1):
        result = measure(fn, inputs, args.samples)
        report["measurements"][name] = result
        record = {"case": name, **result}
        print(json.dumps(record), flush=True)
        with (local / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        tracker.log(
            {
                f"timing/{name}/median_seconds": result["median_seconds"],
                f"timing/{name}/warmup_seconds": result["warmup_compile_and_execution_seconds"],
            },
            step=step,
        )
        (destination / "report.json").write_text(json.dumps(report, indent=2))
        (destination / "metrics.jsonl").write_text((local / "metrics.jsonl").read_text())
        gc.collect()
    # Validate fused versus separately computed outputs on the small end-to-end check.
    if args.smoke:
        fused = jax.block_until_ready(full_fn(gradients, states))[0]
        separate = directions(updated, 0.25, **solver_kwargs)
        for a, b in zip(fused, separate, strict=True):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-5)
    # Timing cases retain the refreshed state. Release it before holding two
    # full update outputs for comparison on devices with smaller HBM.
    del cases, updated
    gc.collect()
    if args.eigh_layout != "replicated" or args.eigh_implementation != "default":
        reference_fn = jax.jit(
            full_update,
            in_shardings=(gradient_shardings, state_shardings),
            out_shardings=(gradient_shardings, state_shardings),
        )
        report["reference_measurements"] = measure(reference_fn, (gradients, states), args.samples)
        (destination / "report.json").write_text(json.dumps(report, indent=2))
        reference = jax.block_until_ready(reference_fn(gradients, states))
        candidate = jax.block_until_ready(full_fn(gradients, states))
        report["reference_comparison"] = compare_trees(candidate, reference)
        tracker.log_summary({"diagnostic/reference_comparison_passed": report["reference_comparison"]["passed"]})
        (destination / "report.json").write_text(json.dumps(report, indent=2))
        del reference, candidate
        if args.checkpoint:
            original_direction_fn = jax.jit(
                partial(directions, alpha=0.25),
                in_shardings=(state_shardings,),
                out_shardings=gradient_shardings,
            )
            report["saved_checkpoint_reference_measurements"] = measure(original_direction_fn, (states,), args.samples)
            (destination / "report.json").write_text(json.dumps(report, indent=2))
            original = jax.block_until_ready(original_direction_fn(states))
            proposed = jax.block_until_ready(checkpoint_direction_fn(states))
            report["saved_checkpoint_reference_comparison"] = compare_trees(proposed, original)
            tracker.log_summary(
                {"diagnostic/checkpoint_reference_passed": report["saved_checkpoint_reference_comparison"]["passed"]}
            )
            (destination / "report.json").write_text(json.dumps(report, indent=2))
            del original, proposed
    report["profile"] = profile_once(full_fn, (gradients, states), local / "profile")
    report["device_timeline_available"] = any(
        p["name"].startswith("/device:TPU:") and p["events"] > 0 for f in report["profile"] for p in f["planes"]
    )
    report["device_memory_stats"] = [d.memory_stats() for d in jax.local_devices()]
    for file in (local / "profile").rglob("*.xplane.pb"):
        target = destination / "profile" / file.relative_to(local / "profile").as_posix()
        target.parent.mkdirs()
        target.upload_from(str(file))
    (local / "report.json").write_text(json.dumps(report, indent=2))
    (destination / "report.json").write_text(json.dumps(report, indent=2))
    tracker.log_summary(
        {"diagnostic/completed": 1, "diagnostic/device_timeline_available": report["device_timeline_available"]}
    )
    tracker.finish()
    multihost_utils.sync_global_devices("optimizer-cost-artifacts-written")
    print(
        json.dumps(
            {
                "completed": True,
                "output": str(destination),
                "device_timeline_available": report["device_timeline_available"],
            }
        ),
        flush=True,
    )

    if jax.distributed.is_initialized():
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
