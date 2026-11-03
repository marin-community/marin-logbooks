# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Locate EMA-Gram numerical divergence; no training or performance claim."""

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import time
import uuid
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from iris.jax.init import initialize_jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.optim.ema_gram import GramState, init_gram_state, matrix_parallel_eigh, relative_gram_power
from levanter.optim.util import NEWTON_SCHULZ_COEFFICIENTS, zeropower_via_newtonschulz5
from levanter.tracker.wandb import WandbConfig
from optimizer_cost import NAMES, ROW_SHARDED, directions, full_update, put_global, refresh
from rigging.filesystem.storage_path import StoragePath

REPORT_REQUEST_TIMEOUT = 60


def fingerprint(tree):
    digest = hashlib.sha256()
    for x in jax.tree.leaves(tree):
        a = np.asarray(x)
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    return digest.hexdigest()


def comparison(actual, expected):
    """Compare host arrays at the unchanged gate, accumulating norms in FP64."""
    rows = []
    for index, (a, b) in enumerate(zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True)):
        assert a.shape == b.shape
        maximum, error2, reference2, passed = 0.0, 0.0, 0.0, True
        for aa, bb in zip(a, b, strict=True):
            difference = aa.astype(np.float64) - bb.astype(np.float64)
            maximum = max(maximum, float(np.max(np.abs(difference))))
            error2 += float(np.sum(difference**2))
            reference2 += float(np.sum(bb.astype(np.float64) ** 2))
            passed &= bool(np.all(np.abs(difference) <= 1e-5 + 1e-5 * np.abs(bb.astype(np.float64))))
        rows.append(
            dict(
                leaf=index,
                max_absolute_error=maximum,
                relative_frobenius_error=math.sqrt(error2 / max(reference2, 1e-60)),
                passed=passed,
            )
        )
    return dict(rtol=1e-5, atol=1e-5, passed=all(r["passed"] for r in rows), leaves=rows)


def normalize(gram):
    symmetric = (gram + jnp.swapaxes(gram, -1, -2)) * 0.5
    mean = jnp.trace(symmetric, axis1=-2, axis2=-1) / symmetric.shape[-1]
    return symmetric / jnp.where(mean > 0, mean, 1)[..., None, None]


def reconstruct(vectors, values):
    weights = (jnp.maximum(values, 0) + 1e-4) ** (-0.25)
    return jnp.matmul(
        vectors * weights[..., None, :], jnp.swapaxes(vectors, -1, -2), precision=jax.lax.Precision.HIGHEST
    )


def eigen_metrics(normalized, vectors, values):
    residual = jnp.matmul(normalized, vectors, precision=jax.lax.Precision.HIGHEST) - vectors * values[..., None, :]
    orthogonality = jnp.matmul(jnp.swapaxes(vectors, -1, -2), vectors, precision=jax.lax.Precision.HIGHEST) - jnp.eye(
        values.shape[-1]
    )
    return {
        "relative_residual_per_matrix": (
            jnp.linalg.norm(residual, axis=(-2, -1)) / jnp.maximum(jnp.linalg.norm(normalized, axis=(-2, -1)), 1e-30)
        ),
        "orthogonality_rms_per_matrix": jnp.linalg.norm(orthogonality, axis=(-2, -1)) / math.sqrt(values.shape[-1]),
        "minimum_eigenvalue_per_matrix": jnp.min(values, axis=-1),
        "maximum_eigenvalue_per_matrix": jnp.max(values, axis=-1),
        "negative_eigenvalues_per_matrix": jnp.sum(values < 0, axis=-1),
    }


def inner_product(left, momentum, right):
    return jnp.matmul(
        jnp.matmul(left, momentum, precision=jax.lax.Precision.HIGHEST), right, precision=jax.lax.Precision.HIGHEST
    )


def polar(inner):
    return jax.vmap(partial(zeropower_via_newtonschulz5, steps=5))(inner)


def downstream(left, momentum, right):
    inner = inner_product(left, momentum, right)
    orthogonal = polar(inner)
    return inner, orthogonal, inner_product(left, orthogonal, right)


def polar_iterates(inner):
    """Expose normalization and five iterates with the original arithmetic.

    Extra outputs change compilation. Compare the final iterate against the
    uninstrumented polar call before interpreting any intermediate difference.
    """

    def one_matrix(x):
        norm = jnp.linalg.norm(x)
        x = x / (norm + 1e-7)
        outputs = [norm, x]
        transposed = x.shape[0] > x.shape[1]
        if transposed:
            x = x.T
        if not jax.sharding.get_abstract_mesh().empty:
            x = jax.lax.with_sharding_constraint(x, P(None, ("data", "model")))
        for a, b, c in NEWTON_SCHULZ_COEFFICIENTS["quintic"]:
            gram = x @ x.T
            polynomial = b * gram + c * gram @ gram
            x = a * x + polynomial @ x
            outputs.append(x.T if transposed else x)
        return tuple(outputs)

    return jax.vmap(one_matrix)(inner)


class Recorder:
    """Persist every stage before progressing to another allocation-heavy kernel."""

    def __init__(self, destination, local, tracker, config):
        self.destination, self.local, self.tracker = destination, local, tracker
        self.report = dict(config=config, stages={}, comparisons={})
        self.step = 0
        self.persist()

    def persist(self):
        payload = json.dumps(self.report, indent=2)
        (self.local / "report.json").write_text(payload)
        (self.destination / "report.json").write_text(payload, requests_timeout=REPORT_REQUEST_TIMEOUT)

    def run(self, name, fn, inputs, in_shardings, out_shardings):
        self.report["active_stage"] = name
        self.persist()
        compiled = jax.jit(fn, in_shardings=in_shardings, out_shardings=out_shardings)
        start = time.perf_counter()
        capture = self.report["config"]["polar_compiler"] and (
            name in ("fused_original", "fused_matrix")
            or name.endswith("/downstream")
            or "/identical_inner/" in name
            or "/polar_iterates/" in name
        )
        if capture:
            lowered = compiled.lower(*inputs)
            executable = lowered.compile()
            result = jax.block_until_ready(executable(*inputs))
        else:
            result = jax.block_until_ready(compiled(*inputs))
        elapsed = time.perf_counter() - start
        memory = [d.memory_stats() for d in jax.local_devices()]
        host = jax.device_get(result)
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(host)), name
        del result
        compiled.clear_cache()
        del compiled
        gc.collect()
        self.step += 1
        self.report["stages"][name] = dict(
            compile_and_execution_seconds=elapsed, finite=True, output_sha256=fingerprint(host), device_memory=memory
        )
        if capture:
            artifacts = {}
            for label, source in [("stablehlo", lowered.as_text()), ("optimized_hlo", executable.as_text())]:
                payload = gzip.compress(source.encode(), mtime=0)
                filename = name.replace("/", "__") + "." + label + ".txt.gz"
                (self.local / filename).write_bytes(payload)
                (self.destination / filename).write_bytes(payload, requests_timeout=REPORT_REQUEST_TIMEOUT)
                artifacts[label] = dict(
                    uri=str(self.destination / filename),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    compressed_bytes=len(payload),
                )
            self.report["stages"][name]["compiler_artifacts"] = artifacts
            del lowered, executable
        self.tracker.log({f"stage/{name}/compile_and_execution_seconds": elapsed}, step=self.step)
        self.persist()
        print(json.dumps(dict(stage=name, seconds=elapsed, finite=True)), flush=True)
        return host

    def compare(self, name, actual, expected):
        result = comparison(actual, expected)
        self.report["comparisons"][name] = result
        self.persist()
        print(json.dumps(dict(comparison=name, **result)), flush=True)
        return result["passed"]


def fixed_direction_outputs(rec, label, fn, banks, state_shardings, output_shardings):
    """Compile once, then change only the input bank for four executions."""
    group = f"fixed_direction/{label}"
    rec.report["active_stage"] = group + "/compile"
    rec.persist()
    exemplar = jax.tree.map(
        lambda x, sh: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sh), banks["common"], state_shardings
    )
    compiled = jax.jit(fn, in_shardings=(state_shardings,), out_shardings=output_shardings)
    start = time.perf_counter()
    lowered = compiled.lower(exemplar)
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - start
    artifacts = {}
    for kind, text in [("stablehlo", lowered.as_text()), ("optimized_hlo", executable.as_text())]:
        payload = gzip.compress(text.encode(), mtime=0)
        filename = group.replace("/", "__") + "." + kind + ".txt.gz"
        (rec.local / filename).write_bytes(payload)
        (rec.destination / filename).write_bytes(payload, requests_timeout=REPORT_REQUEST_TIMEOUT)
        artifacts[kind] = dict(
            uri=str(rec.destination / filename),
            sha256=hashlib.sha256(payload).hexdigest(),
            compressed_bytes=len(payload),
        )
    rec.report.setdefault("fixed_executables", {})[label] = dict(
        compile_count=1,
        compile_seconds=compile_seconds,
        compiler_artifacts=artifacts,
        input_shardings=str(state_shardings),
        output_shardings=str(output_shardings),
        invocation_order=["common", "fused_original", "fused_matrix", "common_replay"],
    )
    rec.persist()
    outputs = {}
    for index, bank_name in enumerate(("common", "fused_original", "fused_matrix", "common_replay")):
        bank = banks["common" if bank_name == "common_replay" else bank_name]
        name = group + "/" + bank_name
        rec.report["active_stage"] = name
        rec.persist()
        device_bank = jax.tree.map(put_global, bank, state_shardings)
        jax.block_until_ready(device_bank)
        start = time.perf_counter()
        result = jax.block_until_ready(executable(device_bank))
        elapsed = time.perf_counter() - start
        memory = [d.memory_stats() for d in jax.local_devices()]
        host = jax.device_get(result)
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(host)), name
        del result, device_bank
        gc.collect()
        outputs[bank_name] = host
        rec.step += 1
        rec.report["stages"][name] = dict(
            execution_seconds=elapsed,
            timing_scope="Single execution; no warmup or performance claim; excludes explicit transfer and compilation",
            finite=True,
            output_sha256=fingerprint(host),
            input_sha256=fingerprint(bank),
            executable_group=label,
            invocation_index=index,
            device_memory=memory,
        )
        rec.tracker.log({f"stage/{name}/execution_seconds": elapsed}, step=rec.step)
        rec.persist()
        print(json.dumps(dict(stage=name, finite=True, executable_group=label, invocation_index=index)), flush=True)
    compiled.clear_cache()
    del compiled, lowered, executable
    gc.collect()
    rec.compare(group + "/replay", outputs["common_replay"], outputs["common"])
    rec.report["fixed_executables"][label]["replay_bitwise_equal"] = fingerprint(
        outputs.pop("common_replay")
    ) == fingerprint(outputs["common"])
    rec.persist()
    return outputs


def state_crossfeed(rec, banks, references, matrix_eigh, state_shardings, output_shardings):
    """Compare saved refresh outputs without recompiling for their values."""
    rec.report["state_banks"] = {
        label: dict(sha256=fingerprint(bank), bytes=sum(x.nbytes for x in jax.tree.leaves(bank)))
        for label, bank in banks.items()
    }
    rec.persist()
    all_outputs = {}
    for label, eig_fn in [("original", jax.lax.linalg.eigh), ("matrix", matrix_eigh)]:
        outputs = fixed_direction_outputs(
            rec, label, partial(directions, alpha=0.25, eigh_fn=eig_fn), banks, state_shardings, output_shardings
        )
        all_outputs[label] = outputs
        for left, right in [
            ("fused_original", "common"),
            ("fused_matrix", "common"),
            ("fused_matrix", "fused_original"),
        ]:
            rec.compare(f"crossfeed/{label}/banks/{left}_vs_{right}", outputs[left], outputs[right])
        for bank_name, output in outputs.items():
            for reference_name, reference in references.items():
                rec.compare(f"crossfeed/{label}/{bank_name}/vs_{reference_name}", output, reference)
        for bank_name, bank in banks.items():
            assert fingerprint(bank) == rec.report["state_banks"][bank_name]["sha256"], "Frozen bank changed"
    for bank_name in banks:
        rec.compare(
            f"crossfeed/evaluators/{bank_name}", all_outputs["matrix"][bank_name], all_outputs["original"][bank_name]
        )


def finish(rec, tracker, fused_pass, smoke):
    if smoke:
        assert all(x["passed"] for x in rec.report["comparisons"].values()), "CPU harness comparison failed"
    # Collection completion never means the candidate is accepted for training.
    rec.report["active_stage"] = "completed"
    rec.report["attribution_completed"] = True
    rec.report["candidate_accepted_for_training"] = False
    rec.persist()
    tracker.log_summary(
        {
            "diagnostic/attribution_completed": 1,
            "diagnostic/fused_comparison_passed": fused_pass,
            "diagnostic/candidate_accepted_for_training": False,
        }
    )
    tracker.finish()
    print(json.dumps(dict(completed=True, output=str(rec.destination))), flush=True)
    if jax.distributed.is_initialized():
        jax.distributed.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--logging-check", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--polar-compiler", action="store_true")
    mode.add_argument("--state-crossfeed", action="store_true")
    args = parser.parse_args()
    if not args.smoke:
        initialize_jax()
    devices = jax.devices()
    assert jax.process_count() == 1, "This diagnostic requires one host"
    if not args.smoke:
        assert len(devices) == 8 and all(
            d.platform == "tpu"
            and ("v5" in d.device_kind.lower())
            and ("lite" in d.device_kind.lower() or "v5e" in d.device_kind.lower())
            for d in devices
        ), devices
    identity = "ema-gram-20-attribution-" + uuid.uuid4().hex[:12]
    destination = StoragePath(args.output_prefix) / identity
    destination.mkdirs()
    local = Path("/tmp") / identity
    local.mkdir()
    layers, width = (3, 8) if args.smoke else (11, 1024)
    shapes = [(layers, width, width)] * 4 + [(layers, width * 4, width)] * 2 + [(layers, width, width * 4)]
    config = dict(
        source_commit=os.environ.get("GIT_COMMIT"),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seed=args.seed,
        shapes=shapes,
        names=NAMES,
        smoke=args.smoke,
        polar_compiler=args.polar_compiler or args.state_crossfeed,
        state_crossfeed=args.state_crossfeed,
        polar_iterate_leaf_order=[
            "input_norm",
            "normalized_input",
            "iteration_1",
            "iteration_2",
            "iteration_3",
            "iteration_4",
            "iteration_5",
        ],
        process_count=1,
        devices=[dict(id=d.id, kind=d.device_kind, platform=d.platform) for d in devices],
        output_prefix=str(destination),
        alpha=0.25,
        ridge=1e-4,
        beta1=0.95,
        beta2=0.99,
        polar_steps=5,
        solver="default",
        gcs_report_request_timeout_seconds=REPORT_REQUEST_TIMEOUT,
        rtol=1e-5,
        atol=1e-5,
        limitations=(
            "Synthetic non-training stage attribution. Single calls include compilation; no performance or MFU claim. "
            "Staging changes compilation. Expected candidate mismatches are recorded, never accepted for training."
        ),
    )
    tracker = WandbConfig(
        entity="marin-community",
        project="marin",
        name=identity,
        id=identity,
        group="ema-gram-20-attribution",
        mode="online",
        resume="never",
        background=False,
        save_code=False,
        replicate_path=str(destination),
    ).init(None)
    print(f"W&B: https://wandb.ai/marin-community/marin/runs/{identity}", flush=True)
    tracker.log_hyperparameters(config)
    tracker.log({"logging_check/success": 1}, step=0)
    (destination / "config.json").write_text(json.dumps(config, indent=2), requests_timeout=REPORT_REQUEST_TIMEOUT)
    if args.logging_check:
        tracker.finish()
        return
    rec = Recorder(destination, local, tracker, config)
    mesh = Mesh(np.array(devices), ("data",))
    rep = NamedSharding(mesh, P())
    gs = tuple(NamedSharding(mesh, P(None, "data", None) if n in ROW_SHARDED else P(None, None, "data")) for n in NAMES)
    ss = tuple(GramState(rep, rep, rep) for _ in NAMES)
    matrix_eigh = partial(matrix_parallel_eigh, mesh=mesh, axis_name="data")
    rng = np.random.default_rng(args.seed)
    scale = 0.1 / math.sqrt(sum(math.prod(s) for s in shapes))
    gradients = tuple(
        put_global(rng.standard_normal(s, dtype=np.float32) * scale, sh) for s, sh in zip(shapes, gs, strict=True)
    )
    prior = tuple(
        put_global(rng.standard_normal(s, dtype=np.float32) * scale, sh) for s, sh in zip(shapes, gs, strict=True)
    )
    zero = rec.run("zero_initialize", lambda p: tuple(init_gram_state(g) for g in p), (prior,), (gs,), ss)
    initial = rec.run("initialize", refresh, (prior, zero), (gs, ss), ss)
    del zero
    del prior
    states = jax.tree.map(lambda x: put_global(x, rep), initial)
    del initial
    common = rec.run("common_refresh", refresh, (gradients, states), (gs, ss), ss)
    original = rec.run("fused_original", full_update, (gradients, states), (gs, ss), (gs, ss))
    candidate = rec.run(
        "fused_matrix", partial(full_update, eigh_fn=matrix_eigh), (gradients, states), (gs, ss), (gs, ss)
    )
    fused_pass = rec.compare("fused_candidate_vs_original", candidate, original)
    rec.compare("original_refresh_vs_common", original[1], common)
    rec.compare("candidate_refresh_vs_common", candidate[1], common)
    if args.state_crossfeed:
        banks = dict(common=common, fused_original=original[1], fused_matrix=candidate[1])
        references = dict(fused_original=original[0], fused_matrix=candidate[0])
        del original, candidate, gradients, states
        jax.clear_caches()
        gc.collect()
        state_crossfeed(rec, banks, references, matrix_eigh, ss, gs)
        assert len(rec.report["stages"]) == 13 and len(rec.report["comparisons"]) == 26
        finish(rec, tracker, fused_pass, args.smoke)
        return
    original_directions, candidate_directions = original[0], candidate[0]
    del original, candidate, gradients, states
    gc.collect()
    rec.report["common_state_sha256"] = fingerprint(common)
    rec.persist()
    for name, state, sharding, orig_dir, cand_dir in zip(
        NAMES, common, gs, original_directions, candidate_directions, strict=True
    ):
        power_banks = {"original": [], "matrix": []}
        for side, gram in [("left", state.left), ("right", state.right)]:
            normalized = rec.run(f"{name}/{side}/normalize", normalize, (gram,), (rep,), rep)
            values_bank = {}
            for label, eig_fn in [("original", jax.lax.linalg.eigh), ("matrix", matrix_eigh)]:
                vectors, values = rec.run(
                    f"{name}/{side}/{label}/eigh",
                    partial(eig_fn, implementation=None),
                    (normalized,),
                    (rep,),
                    (rep, rep),
                )
                metrics = rec.run(
                    f"{name}/{side}/{label}/residual", eigen_metrics, (normalized, vectors, values), (rep, rep, rep), rep
                )
                rec.report["stages"][f"{name}/{side}/{label}/residual"]["metrics"] = {
                    k: v.tolist() for k, v in metrics.items()
                }
                power = rec.run(f"{name}/{side}/{label}/reconstruct", reconstruct, (vectors, values), (rep, rep), rep)
                del vectors, metrics
                fused_power = rec.run(
                    f"{name}/{side}/{label}/fused_power",
                    partial(relative_gram_power, alpha=0.25, ridge=1e-4, eigh_fn=eig_fn),
                    (gram,),
                    (rep,),
                    rep,
                )
                rec.compare(f"{name}/{side}/{label}/staged_vs_fused_power", power, fused_power)
                del fused_power
                values_bank[label] = values
                power_banks[label].append(power)
            rec.compare(f"{name}/{side}/eigenvalues", values_bank["matrix"], values_bank["original"])
            rec.compare(f"{name}/{side}/powers", power_banks["matrix"][-1], power_banks["original"][-1])
            del values_bank, normalized, power, values
        downstream_banks = {}
        for label in ("original", "matrix"):
            left, right = power_banks[label]
            for layout, output_sharding in [("replicated", rep), ("direction_sharded", sharding)]:
                key = f"{label}/{layout}"
                downstream_banks[key] = rec.run(
                    f"{name}/{key}/downstream",
                    downstream,
                    (left, state.momentum, right),
                    (rep, rep, rep),
                    (output_sharding,) * 3,
                )
            rec.compare(
                f"{name}/{label}/identical_powers_layout_control",
                downstream_banks[f"{label}/direction_sharded"],
                downstream_banks[f"{label}/replicated"],
            )
            rec.compare(
                f"{name}/{label}/staged_direction_vs_fused",
                downstream_banks[f"{label}/direction_sharded"][2],
                orig_dir if label == "original" else cand_dir,
            )
        rec.compare(
            f"{name}/power_branch_downstream",
            downstream_banks["matrix/replicated"],
            downstream_banks["original/replicated"],
        )
        common_inner = downstream_banks["original/replicated"][0]
        polar_rep = rec.run(f"{name}/identical_inner/replicated", polar, (common_inner,), (rep,), rep)
        polar_shard = rec.run(f"{name}/identical_inner/direction_sharded", polar, (common_inner,), (rep,), sharding)
        rec.compare(f"{name}/identical_inner_layout_control", polar_shard, polar_rep)
        if args.polar_compiler:
            iterates = {}
            for layout, output_sharding in [("replicated", rep), ("direction_sharded", sharding)]:
                iterates[layout] = rec.run(
                    f"{name}/polar_iterates/{layout}",
                    polar_iterates,
                    (common_inner,),
                    (rep,),
                    (rep,) + (output_sharding,) * 6,
                )
                rec.compare(
                    f"{name}/polar_iterates/{layout}/final_vs_uninstrumented",
                    iterates[layout][-1],
                    polar_rep if layout == "replicated" else polar_shard,
                )
            rec.compare(f"{name}/polar_iterates/layout_control", iterates["direction_sharded"], iterates["replicated"])
            del iterates
        assert rec.report["common_state_sha256"] == fingerprint(common), "Frozen host inputs changed"
        del power_banks, downstream_banks, left, right, common_inner, polar_rep, polar_shard
        gc.collect()
    finish(rec, tracker, fused_pass, args.smoke)


if __name__ == "__main__":
    main()
