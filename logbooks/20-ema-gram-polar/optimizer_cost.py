# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure EMA-Gram kernels on synthetic inputs with the pilot's matrix geometry.

This is a non-training diagnostic. It does not measure language-model quality
or reproduce the evolving spectrum of a trained checkpoint.
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
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.optim.ema_gram import GramState, gram_direction, init_gram_state, relative_gram_power, update_gram_state
from levanter.tracker.wandb import WandbConfig
from marin.profiling.xplane import _xspace_message_class
from rigging.filesystem.storage_path import StoragePath

NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ROW_SHARDED = {"o_proj", "down_proj"}


def refresh(gradients, states):
    return tuple(update_gram_state(g, s, 0.95, 0.99) for g, s in zip(gradients, states, strict=True))


def powers(states):
    return tuple((relative_gram_power(s.left, 0.25, 1e-4), relative_gram_power(s.right, 0.25, 1e-4)) for s in states)


def directions(states, alpha):
    return tuple(gram_direction(s, alpha=alpha, ridge=1e-4, polar_steps=5) for s in states)


def full_update(gradients, states):
    updated = refresh(gradients, states)
    return directions(updated, 0.25), updated


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
    parser.add_argument("--smoke", action="store_true", help="Small CPU geometry; never performance evidence")
    parser.add_argument("--logging-check", action="store_true", help="Online tracker only; no kernel execution")
    args = parser.parse_args()
    assert args.samples > 0
    mode = "logging-check" if args.logging_check else "cpu-smoke" if args.smoke else "tpu-cost"
    identity = f"ema-gram-20-{mode}-{uuid.uuid4().hex[:12]}"
    local = Path("/tmp") / identity
    local.mkdir(parents=True)
    destination = StoragePath(args.output_prefix) / identity
    destination.mkdirs()
    layers, width = (3, 8) if args.smoke else (11, 1024)
    shapes = [(layers, width, width)] * 4 + [(layers, width * 4, width)] * 2 + [(layers, width, width * 4)]
    config = {
        "mode": mode,
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
        "input_state": "one synthetic gradient update from zero; same inputs for every timing",
        "limitations": (
            "No model/data/AdamW/weight decay. Synthetic spectra; "
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
        replicate_path=str(destination),
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
        assert len(devices) == 4 and all(d.platform == "tpu" and "v4" in d.device_kind.lower() for d in devices), devices
    mesh = Mesh(np.array(devices), ("data",))
    replicated = NamedSharding(mesh, P())
    gradient_shardings = tuple(
        NamedSharding(mesh, P(None, "data", None) if n in ROW_SHARDED else P(None, None, "data")) for n in NAMES
    )
    state_shardings = tuple(GramState(replicated, replicated, replicated) for _ in NAMES)
    config["devices"] = [{"id": d.id, "platform": d.platform, "kind": d.device_kind} for d in devices]
    tracker.log_hyperparameters(config)
    rng = np.random.default_rng(args.seed)
    scale = 0.1 / math.sqrt(sum(math.prod(shape) for shape in shapes))
    gradients = tuple(
        jax.device_put(rng.standard_normal(shape, dtype=np.float32) * scale, sharding)
        for shape, sharding in zip(shapes, gradient_shardings, strict=True)
    )
    prior = tuple(
        jax.device_put(rng.standard_normal(shape, dtype=np.float32) * scale, sharding)
        for shape, sharding in zip(shapes, gradient_shardings, strict=True)
    )
    initialize = jax.jit(lambda gs: tuple(init_gram_state(g) for g in gs), out_shardings=state_shardings)
    refresh_fn = jax.jit(refresh, in_shardings=(gradient_shardings, state_shardings), out_shardings=state_shardings)
    states = jax.block_until_ready(refresh_fn(prior, initialize(prior)))
    del prior
    updated = jax.block_until_ready(refresh_fn(gradients, states))
    full_fn = jax.jit(
        full_update,
        in_shardings=(gradient_shardings, state_shardings),
        out_shardings=(gradient_shardings, state_shardings),
    )
    cases = [
        ("gram_refresh", refresh_fn, (gradients, states)),
        ("spectral_powers", jax.jit(powers, in_shardings=(state_shardings,), out_shardings=replicated), (updated,)),
        (
            "alpha0_direction",
            jax.jit(partial(directions, alpha=0), in_shardings=(state_shardings,), out_shardings=gradient_shardings),
            (updated,),
        ),
        ("quarter_update_and_direction", full_fn, (gradients, states)),
    ]
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
        separate = directions(updated, 0.25)
        for a, b in zip(fused, separate, strict=True):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-5)
    report["profile"] = profile_once(full_fn, (gradients, states), local / "profile")
    report["device_timeline_available"] = any(
        p["name"].startswith("/device:TPU:") for f in report["profile"] for p in f["planes"]
    )
    report["device_memory_stats"] = [d.memory_stats() for d in devices]
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


if __name__ == "__main__":
    main()
