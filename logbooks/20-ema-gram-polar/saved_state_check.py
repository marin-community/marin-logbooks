# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finite/immutability checks on saved EMA-Gram state; equivalence is diagnostic only."""

import argparse
import gc
import hashlib
import json
import math
import os
import tempfile
import time
import uuid
from functools import partial
from pathlib import Path

import jax
import numpy as np
from iris.jax.init import initialize_jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import save_checkpoint
from levanter.optim.ema_gram import EmaGramState, GramState, matrix_parallel_eigh
from levanter.tracker.wandb import WandbConfig
from optimizer_cost import (
    NAMES,
    ROW_SHARDED,
    checkpoint_states,
    directions,
    full_update,
    init_gram_state,
    put_global,
    refresh,
)
from rigging.filesystem.storage_path import StoragePath


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def snapshot(tree):
    """Read local shards, checking finite values and exact agreement of local replicas."""
    result = {}
    for leaf, array in enumerate(jax.tree.leaves(tree)):
        for shard in array.addressable_shards:
            value = np.asarray(shard.data)
            assert np.isfinite(value).all(), f"Nonfinite leaf {leaf} on device {shard.device.id}"
            key = (leaf, str(shard.index))
            if key in result:
                assert result[key].tobytes() == value.tobytes(), f"Local replicas differ: {key}"
            else:
                result[key] = value.copy()
    return result


def fingerprints(tree):
    arrays = snapshot(tree)
    return {key: sha(value.tobytes()) for key, value in arrays.items()}


def compare(actual, reference):
    """Local unique path/index statistics; deduplicate across hosts before global norms."""
    assert actual.keys() == reference.keys()
    rows = []
    for (leaf, index), a in actual.items():
        b = reference[(leaf, index)]
        assert a.shape == b.shape and a.dtype == b.dtype
        # Host diagnostic accumulation only; kernel inputs and arithmetic are unchanged.
        aa, bb = a.astype(np.float64), b.astype(np.float64)
        delta = aa - bb
        rows.append(
            dict(
                leaf=leaf,
                index=index,
                shape=list(a.shape),
                dtype=str(a.dtype),
                max_absolute_error=float(np.abs(delta).max()),
                squared_error=float(np.square(delta).sum()),
                squared_reference=float(np.square(bb).sum()),
                passed=bool(np.all(np.abs(delta) <= 1e-5 + 1e-5 * np.abs(bb))),
                actual_sha256=sha(a.tobytes()),
                reference_sha256=sha(b.tobytes()),
            )
        )
    return dict(
        rtol=1e-5,
        atol=1e-5,
        local_passed=all(r["passed"] for r in rows),
        shards=rows,
        scope="Local unique leaf/index records; not a global gate until all process reports are joined",
    )


def source_files():
    base = Path(__file__).resolve().parent
    root = base.parent.parent
    paths = [Path(__file__).resolve(), base / "optimizer_cost.py", root / "lib/levanter/src/levanter/optim/ema_gram.py"]
    return {str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes())) for p in paths}


def execute(args, tracker, persist):
    cpu = args.mode == "cpu-check"
    assert jax.default_backend() == ("cpu" if cpu else "tpu")
    if not cpu:
        assert jax.device_count() == 16 and jax.process_count() == 4
        assert all("v4" in d.device_kind.lower() for d in jax.devices())
    mesh = Mesh(np.array(jax.devices()), ("data",))
    replicated = NamedSharding(mesh, P())
    gradient_shardings = tuple(
        NamedSharding(mesh, P(None, "data", None) if n in ROW_SHARDED else P(None, None, "data")) for n in NAMES
    )
    state_shardings = tuple(GramState(replicated, replicated, replicated) for _ in NAMES)
    layers, width = (3, 8) if cpu else (11, 1024)
    shapes = [(layers, width, width)] * 4 + [(layers, width * 4, width)] * 2 + [(layers, width, width * 4)]
    rng = np.random.default_rng(0)
    scale = 0.1 / math.sqrt(sum(math.prod(s) for s in shapes))
    gradients = tuple(
        put_global(rng.standard_normal(s, dtype=np.float32) * scale, sh)
        for s, sh in zip(shapes, gradient_shardings, strict=True)
    )
    fixture = tempfile.TemporaryDirectory(prefix="ema-saved-state-") if cpu else None
    try:
        if cpu:
            states = jax.block_until_ready(
                jax.jit(refresh, out_shardings=state_shardings)(gradients, tuple(init_gram_state(g) for g in gradients))
            )
            expected = fingerprints(states)
            checkpoint = str(Path(fixture.name) / "step-0")
            save_checkpoint({"saved": EmaGramState(states)}, 0, checkpoint)
            del states
        else:
            checkpoint = args.checkpoint
            identities = json.loads((Path(__file__).parent / "loss-pair-checkpoint-metadata.json").read_text())
            assert checkpoint == identities["checkpoint"]
            for name, expected_file in identities["files"].items():
                raw = (StoragePath(checkpoint) / name).read_bytes()
                assert len(raw) == expected_file["bytes"] and sha(raw) == expected_file["sha256"]
        states, evidence = checkpoint_states(checkpoint, shapes, mesh)
        before = fingerprints(states)
        if cpu:
            assert before == expected, "Generated checkpoint state did not round-trip"
        report = dict(
            mode=args.mode,
            process_index=jax.process_index(),
            process_count=jax.process_count(),
            source_files=source_files(),
            checkpoint=evidence,
            shapes=shapes,
            gradient_seed=0,
            gradient_scale=scale,
            input_state_finite=True,
            cases={},
            training_steps_executed=0,
            full_training_memory_fit_verified=False,
            full_trainer_restore=False,
            equivalence_is_training_veto=False,
            scope="Saved Gram/momentum only, synthetic gradients, no model/data/AdamW/training MFU",
        )
        persist(report)
        matrix_eigh = partial(matrix_parallel_eigh, mesh=mesh, axis_name="data")
        for case, full in [("unchanged_saved_state_direction", False), ("fixed_synthetic_gradient_update", True)]:
            inputs = (gradients, states) if full else (states,)
            in_shardings = (gradient_shardings, state_shardings) if full else (state_shardings,)
            out_shardings = (gradient_shardings, state_shardings) if full else gradient_shardings
            reference = None
            results = {}
            for layout, eig in [("replicated", jax.lax.linalg.eigh), ("matrix", matrix_eigh)]:
                function = partial(full_update, eigh_fn=eig) if full else partial(directions, alpha=0.25, eigh_fn=eig)
                fn = jax.jit(function, in_shardings=in_shardings, out_shardings=out_shardings)
                tracker.log({f"{case}/{layout}/started": 1}, step=len(report["cases"]) * 2 + (layout == "matrix"))
                begin = time.perf_counter()
                output = jax.block_until_ready(fn(*inputs))
                elapsed = time.perf_counter() - begin
                arrays = snapshot(output)
                assert len({leaf for leaf, _ in arrays}) == (28 if full else 7)
                del output
                fn.clear_cache()
                del fn
                gc.collect()
                assert fingerprints(states) == before, "Saved input state changed"
                results[layout] = dict(
                    all_local_outputs_finite=True, input_state_unchanged=True, seconds_including_compilation=elapsed
                )
                if layout == "replicated":
                    reference = arrays
                else:
                    results["comparison"] = compare(arrays, reference)
                    del arrays
            report["cases"][case] = results
            persist(report)
            del reference
            gc.collect()
        if cpu:
            arrays = snapshot(gradients)
            altered = {k: v.copy() for k, v in arrays.items()}
            altered[next(iter(altered))].flat[0] += 1.0
            assert not compare(arrays, altered)["local_passed"]
            assert compare(arrays, arrays)["local_passed"]
            report["cpu_fixture"] = dict(
                checkpoint_round_trip=True,
                finite_mismatch_recorded_without_veto=True,
                comparison_continues_after_mismatch=True,
            )
        report["completed"] = True
        persist(report)
        tracker.log({"saved_state_check/success": 1}, step=4)
        return report
    finally:
        if fixture is not None:
            fixture.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["logging-check", "cpu-check", "worker"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--source-sha256")
    parser.add_argument("--source-manifest", type=Path)
    args = parser.parse_args()
    if args.mode == "worker":
        assert args.source_sha256 == sha(Path(__file__).read_bytes())
        assert args.checkpoint and args.source_manifest
        assert json.loads(args.source_manifest.read_text())["files"] == source_files()
        initialize_jax()
    identity = "ema-gram-saved-state-" + args.mode + "-" + uuid.uuid4().hex[:12]
    destination = StoragePath(args.output_prefix) / identity
    destination.mkdirs()
    tracker = WandbConfig(
        entity="marin-community",
        project="marin",
        id=identity,
        name=identity,
        group="ema-gram-20-saved-state",
        mode="online",
        resume="never",
        background=False,
        save_code=False,
        replicate_path=str(destination),
    ).init(None)
    print(f"W&B: https://wandb.ai/marin-community/marin/runs/{identity}", flush=True)
    config = dict(
        mode=args.mode,
        identity=identity,
        source_files=source_files(),
        source_commit=os.environ.get("GIT_COMMIT"),
        process_index=jax.process_index(),
    )
    tracker.log_hyperparameters(config)
    tracker.log({"logging_check/success": 1}, step=0)
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    try:
        if args.mode == "logging-check":
            return

        def persist(report):
            (destination / "report.json").write_text(json.dumps(report, indent=2))

        execute(args, tracker, persist)
        print(
            json.dumps({"report": str(destination / "report.json"), "process": jax.process_index(), "success": True}),
            flush=True,
        )
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
