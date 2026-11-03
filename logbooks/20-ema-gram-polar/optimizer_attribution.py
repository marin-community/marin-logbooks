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
from optimizer_input_capture import capture_program
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


def polar_products(inner):
    """Expose first-iteration products; extra outputs can change compilation."""

    def one_matrix(x):
        # This diagnostic is restricted to the square query/key/value matrices.
        assert x.shape[0] == x.shape[1]
        norm = jnp.linalg.norm(x)
        x = x / (norm + 1e-7)
        if not jax.sharding.get_abstract_mesh().empty:
            x = jax.lax.with_sharding_constraint(x, P(None, ("data", "model")))
        result = dict(norm=norm, normalized_input=x)
        for iteration, (a, b, c) in enumerate(NEWTON_SCHULZ_COEFFICIENTS["quintic"]):
            gram = x @ x.T
            cubic = (c * gram) @ gram
            polynomial = b * gram + cubic
            product = polynomial @ x
            x = a * x + product
            if iteration == 0:
                result.update(A=gram, C=cubic, B=polynomial, D=product, X1=x)
        result["final"] = x
        return result

    return jax.vmap(one_matrix)(inner)


def isolated_product(stage, *operands):
    a, b, c = NEWTON_SCHULZ_COEFFICIENTS["quintic"][0]
    if stage == "A":
        return operands[0] @ jnp.swapaxes(operands[0], -1, -2)
    if stage == "C":
        return (c * operands[0]) @ operands[0]
    if stage == "B":
        return b * operands[0] + operands[1]
    if stage == "D":
        return operands[0] @ operands[1]
    if stage == "X1":
        return a * operands[0] + operands[1]
    raise ValueError(stage)


def contracting_product(x, sharding):
    """Request column-sharded operands without changing product math or precision."""
    x = jax.lax.with_sharding_constraint(x, sharding)
    return isolated_product("A", x)


def product_crossfeed(rec, name, stage, traces, rep, sharding):
    """Reuse one executable per layout across both frozen operand banks and replay."""
    keys = {
        "A": ("normalized_input",),
        "C": ("A",),
        "B": ("A", "C"),
        "D": ("B", "normalized_input"),
        "X1": ("normalized_input", "D"),
    }[stage]
    banks = {layout: tuple(trace[key] for key in keys) for layout, trace in traces.items()}
    inputs_sha = {label: fingerprint(bank) for label, bank in banks.items()}
    outputs = {}
    layouts = [("replicated", rep), ("direction_sharded", sharding)]
    if rec.report["config"].get("contracting_control"):
        assert stage == "A", "The contracting-dimension control is defined only for A"
        layouts.append(("contracting_input", sharding))
    for layout, out_sharding in layouts:
        group = f"{name}/fixed_product/{stage}/{layout}"
        fn = (
            partial(contracting_product, sharding=sharding)
            if layout == "contracting_input"
            else partial(isolated_product, stage)
        )
        compiled = jax.jit(fn, in_shardings=(rep,) * len(keys), out_shardings=out_sharding)
        rec.report["active_stage"] = group + "/compile"
        rec.persist()
        exemplar = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=rep) for x in banks["replicated"])
        lowered = compiled.lower(*exemplar)
        executable = lowered.compile()
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
        rec.report.setdefault("fixed_product_executables", {})[group] = dict(
            compile_count=1,
            compiler_artifacts=artifacts,
            operands=keys,
            input_sha256=inputs_sha,
            input_shardings=str((rep,) * len(keys)),
            output_sharding=str(out_sharding),
            input_scope="Frozen instrumented operands; see product_decisions for original-path eligibility",
            requested_internal_sharding=str(sharding) if layout == "contracting_input" else None,
            internal_layout_verification="Inspect saved optimized HLO; a requested constraint is not verification",
        )
        outputs[layout] = {}
        for bank_name in ("replicated", "direction_sharded", "replicated_replay"):
            bank = banks["replicated" if bank_name == "replicated_replay" else bank_name]
            invocation = group + "/" + bank_name
            rec.report["active_stage"] = invocation
            rec.persist()
            device_bank = tuple(put_global(x, rep) for x in bank)
            result = jax.block_until_ready(executable(*device_bank))
            host = jax.device_get(result)
            assert np.isfinite(host).all(), invocation
            memory = [d.memory_stats() for d in jax.local_devices()]
            del result, device_bank
            outputs[layout][bank_name] = host
            rec.step += 1
            rec.report["stages"][invocation] = dict(
                finite=True, input_sha256=fingerprint(bank), output_sha256=fingerprint(host), device_memory=memory
            )
            rec.tracker.log({f"stage/{invocation}/finite": 1}, step=rec.step)
            rec.persist()
        rec.compare(group + "/replay", outputs[layout]["replicated_replay"], outputs[layout]["replicated"])
        rec.report["fixed_product_executables"][group]["replay_bitwise_equal"] = fingerprint(
            outputs[layout].pop("replicated_replay")
        ) == fingerprint(outputs[layout]["replicated"])
        rec.compare(group + "/banks", outputs[layout]["direction_sharded"], outputs[layout]["replicated"])
        for bank_name in banks:
            rec.compare(group + "/" + bank_name + "/vs_traced", outputs[layout][bank_name], traces[bank_name][stage])
        compiled.clear_cache()
        del compiled, lowered, executable
        gc.collect()
    for bank_name in banks:
        rec.compare(
            f"{name}/fixed_product/{stage}/{bank_name}/layouts",
            outputs["direction_sharded"][bank_name],
            outputs["replicated"][bank_name],
        )
        assert fingerprint(banks[bank_name]) == inputs_sha[bank_name]
        if "contracting_input" in outputs:
            for baseline in ("replicated", "direction_sharded"):
                rec.compare(
                    f"{name}/fixed_product/{stage}/{bank_name}/contracting_vs_{baseline}",
                    outputs["contracting_input"][bank_name],
                    outputs[baseline][bank_name],
                )
    rec.persist()
    return outputs


def polar_continuation(bank):
    """Continue five polar iterations from normalized X and an external first Gram."""

    def one_matrix(x, gram):
        assert x.shape[0] == x.shape[1]
        if not jax.sharding.get_abstract_mesh().empty:
            x = jax.lax.with_sharding_constraint(x, P(None, ("data", "model")))
        for iteration, (a, b, c) in enumerate(NEWTON_SCHULZ_COEFFICIENTS["quintic"]):
            if iteration:
                gram = x @ x.T
            polynomial = b * gram + (c * gram) @ gram
            x = a * x + polynomial @ x
        return x

    return jax.vmap(one_matrix)(*bank)


def normalized_polar_input(inner):
    """Use the shared polar function's normalization and constraint without iterations."""
    return jax.vmap(partial(zeropower_via_newtonschulz5, steps=0))(inner)


def normalization_first_gram(inner):
    def one_matrix(x):
        assert x.shape[0] == x.shape[1]
        x = zeropower_via_newtonschulz5(x, steps=0)
        return x @ x.T

    return jax.vmap(one_matrix)(inner)


def normalized_first_gram(inner):
    def one_matrix(x):
        assert x.shape[0] == x.shape[1]
        if not jax.sharding.get_abstract_mesh().empty:
            x = jax.lax.with_sharding_constraint(x, P(None, ("data", "model")))
        return x @ x.T

    return jax.vmap(one_matrix)(inner)


def polar_from_normalized(inner):
    """Retain original square-matrix iterations, omitting only normalization."""

    def one_matrix(x):
        assert x.shape[0] == x.shape[1]
        if not jax.sharding.get_abstract_mesh().empty:
            x = jax.lax.with_sharding_constraint(x, P(None, ("data", "model")))
        for a, b, c in NEWTON_SCHULZ_COEFFICIENTS["quintic"]:
            gram = x @ x.T
            polynomial = b * gram + c * gram @ gram
            x = a * x + polynomial @ x
        return x

    return jax.vmap(one_matrix)(inner)


def compare_bitwise(rec, name, actual, expected):
    passed = rec.compare(name, actual, expected)
    rec.report["comparisons"][name]["bitwise_equal"] = fingerprint(actual) == fingerprint(expected)
    rec.persist()
    return passed


def normalization_boundary(rec, name, inner, rep, sharding):
    normalized = rec.run(f"{name}/normalization/input", normalized_polar_input, (inner,), (rep,), rep)
    raw_hash, normalized_hash = fingerprint(inner), fingerprint(normalized)
    references, outputs = {}, {}
    for layout, out_sharding in [("replicated", rep), ("direction_sharded", sharding)]:
        references[layout] = rec.run(f"{name}/identical_inner/{layout}", polar, (inner,), (rep,), out_sharding)
        outputs[layout] = {}
        for mode, fn, bank in (
            ("raw_first_A", normalization_first_gram, inner),
            ("normalized_first_A", normalized_first_gram, normalized),
            ("normalized_polar", polar_from_normalized, normalized),
        ):
            label = f"{name}/normalization/{mode}/{layout}"
            values = fixed_bank_outputs(rec, label, label, fn, {"input": bank}, rep, out_sharding)
            outputs[layout][mode] = values["input"]
        compare_bitwise(
            rec,
            f"{name}/normalization/{layout}/A_boundary",
            outputs[layout]["normalized_first_A"],
            outputs[layout]["raw_first_A"],
        )
    for mode in ("raw_first_A", "normalized_first_A", "normalized_polar"):
        compare_bitwise(
            rec, f"{name}/normalization/{mode}/layouts", outputs["direction_sharded"][mode], outputs["replicated"][mode]
        )
    compare_bitwise(
        rec, f"{name}/normalization/reference/layouts", references["direction_sharded"], references["replicated"]
    )
    checks = {}
    for layout in outputs:
        for reference_layout in references:
            label = f"{name}/normalization/{layout}/polar_vs_reference_{reference_layout}"
            passed = compare_bitwise(rec, label, outputs[layout]["normalized_polar"], references[reference_layout])
            checks[label] = dict(passed=passed, bitwise_equal=rec.report["comparisons"][label]["bitwise_equal"])
    assert fingerprint(inner) == raw_hash and fingerprint(normalized) == normalized_hash
    rec.report.setdefault("normalization_boundary_checks", {})[name] = dict(
        raw_input_sha256=raw_hash,
        normalized_input_sha256=normalized_hash,
        reference_checks=checks,
        original_path_interpretation_allowed=False,
        limitation=(
            "Separating normalization and returning only a first Gram changes compilation. "
            "Final agreement does not prove equal internal values or isolate arithmetic precision."
        ),
    )
    rec.persist()


def raw_contracting_gram(inner, sharding):
    """Normalize through the shared implementation, then constrain product operands."""
    return contracting_product(normalized_polar_input(inner), sharding)


def normalization_partition(rec, name, inner, rep, sharding):
    """Pair normalization boundaries with the same requested contraction partition."""
    normalized = rec.run(f"{name}/partition/input", normalized_polar_input, (inner,), (rep,), rep)
    raw_hash, normalized_hash = fingerprint(inner), fingerprint(normalized)
    internal = NamedSharding(rep.mesh, P(None, None, "data"))
    layouts = [("replicated", rep), ("direction_sharded", sharding)]
    references, products = {}, {}
    for layout, out_sharding in layouts:
        references[layout] = rec.run(f"{name}/identical_inner/{layout}", polar, (inner,), (rep,), out_sharding)
        products[layout] = {}
        for boundary, fn, bank in (
            ("raw", partial(raw_contracting_gram, sharding=internal), inner),
            ("normalized", partial(contracting_product, sharding=internal), normalized),
        ):
            label = f"{name}/partition/{boundary}_first_A/{layout}"
            result = fixed_bank_outputs(rec, label, label, fn, {"input": bank}, rep, out_sharding)
            products[layout][boundary] = result["input"]
            rec.report["fixed_executables"][label].update(
                requested_internal_sharding=str(internal),
                internal_layout_verification="Inspect output ancestry in optimized HLO; constraint alone is not proof",
            )
        compare_bitwise(
            rec, f"{name}/partition/{layout}/A_boundary", products[layout]["normalized"], products[layout]["raw"]
        )
    for boundary in ("raw", "normalized"):
        compare_bitwise(
            rec,
            f"{name}/partition/{boundary}/A_layouts",
            products["direction_sharded"][boundary],
            products["replicated"][boundary],
        )
    compare_bitwise(
        rec, f"{name}/partition/reference/layouts", references["direction_sharded"], references["replicated"]
    )
    banks = {
        f"{layout}_{boundary}_A": (normalized, products[layout][boundary])
        for layout, _ in layouts
        for boundary in ("raw", "normalized")
    }
    rec.report.setdefault("partition_banks", {})[name] = {
        key: dict(sha256=fingerprint(bank), x_sha256=fingerprint(bank[0]), A_sha256=fingerprint(bank[1]))
        for key, bank in banks.items()
    }
    outputs, checks = {}, {}
    first = next(iter(banks))
    for layout, out_sharding in layouts:
        label = f"{name}/partition/continuation/{layout}"
        outputs[layout] = fixed_bank_outputs(rec, label, label, polar_continuation, banks, (rep, rep), out_sharding)
        for bank_name, result in outputs[layout].items():
            if bank_name != first:
                compare_bitwise(rec, f"{label}/{bank_name}/vs_first_bank", result, outputs[layout][first])
            for reference_layout, reference in references.items():
                key = f"{label}/{bank_name}/vs_reference_{reference_layout}"
                passed = compare_bitwise(rec, key, result, reference)
                checks[key] = dict(passed=passed, bitwise_equal=rec.report["comparisons"][key]["bitwise_equal"])
    for bank_name in banks:
        compare_bitwise(
            rec,
            f"{name}/partition/{bank_name}/continuation_layouts",
            outputs["direction_sharded"][bank_name],
            outputs["replicated"][bank_name],
        )
    assert fingerprint(inner) == raw_hash and fingerprint(normalized) == normalized_hash
    rec.report.setdefault("normalization_partition_checks", {})[name] = dict(
        raw_input_sha256=raw_hash,
        normalized_input_sha256=normalized_hash,
        reference_checks=checks,
        original_path_interpretation_allowed=False,
        limitation=(
            "Constraints and separate boundaries can change casts, kernels and supplied values. "
            "Conditional sensitivity does not isolate collective order or precision. "
            "Final agreement cannot establish internal identity or optimizer acceptance."
        ),
    )
    rec.persist()


def partial_gram_function(mesh):
    """Materialize each device's unreduced local product in a leading bank axis."""

    def local_product(x):
        return isolated_product("A", x)[None, ...]

    return jax.shard_map(
        local_product,
        mesh=mesh,
        axis_names={"data"},
        in_specs=P(None, None, "data"),
        out_specs=P("data", None, None, None),
    )


def reduce_partial_function(mesh, scatter):
    """Apply an explicit collective to the same frozen local-product bank."""

    def reduce_local(bank):
        assert bank.shape[0] == 1
        local = bank[0]
        if scatter:
            return jax.lax.psum_scatter(local, "data", scatter_dimension=2, tiled=True)
        return jax.lax.psum(local, "data")

    return jax.shard_map(
        reduce_local,
        mesh=mesh,
        axis_names={"data"},
        in_specs=P("data", None, None, None),
        out_specs=P(None, None, "data") if scatter else P(),
    )


def frozen_partial_collectives(rec, name, inner, rep, sharding):
    """Hold per-device partial values fixed across two collective executables."""
    normalized = rec.run(f"{name}/collectives/input", normalized_polar_input, (inner,), (rep,), rep)
    normalized_hash = fingerprint(normalized)
    partial_sharding = NamedSharding(rep.mesh, P("data", None, None, None))
    label = f"{name}/collectives/local_partials"
    partials = fixed_bank_outputs(
        rec, label, label, partial_gram_function(rep.mesh), {"input": normalized}, rep, partial_sharding
    )["input"]
    assert partials.shape == (rep.mesh.size, *normalized.shape)
    partial_hash = fingerprint(partials)
    rec.report.setdefault("partial_banks", {})[name] = dict(
        shape=list(partials.shape),
        dtype=str(partials.dtype),
        sha256=partial_hash,
        per_device_sha256=[fingerprint(x) for x in partials],
        normalized_input_sha256=normalized_hash,
    )
    products, references, raw_products = {}, {}, {}
    layouts = [("replicated", rep), ("direction_sharded", sharding)]
    internal = NamedSharding(rep.mesh, P(None, None, "data"))
    for layout, out_sharding in layouts:
        label = f"{name}/collectives/reduce/{layout}"
        products[layout] = fixed_bank_outputs(
            rec,
            label,
            label,
            reduce_partial_function(rep.mesh, layout == "direction_sharded"),
            {"partials": partials},
            partial_sharding,
            out_sharding,
        )["partials"]
        references[layout] = rec.run(f"{name}/identical_inner/{layout}", polar, (inner,), (rep,), out_sharding)
        raw_products[layout] = rec.run(
            f"{name}/collectives/raw_first_A/{layout}",
            partial(raw_contracting_gram, sharding=internal),
            (inner,),
            (rep,),
            out_sharding,
        )
    compare_bitwise(rec, f"{name}/collectives/A_layouts", products["direction_sharded"], products["replicated"])
    compare_bitwise(
        rec, f"{name}/collectives/reference_layouts", references["direction_sharded"], references["replicated"]
    )
    for layout in products:
        for ref_layout in raw_products:
            compare_bitwise(
                rec, f"{name}/collectives/{layout}/A_vs_raw_{ref_layout}", products[layout], raw_products[ref_layout]
            )
    banks = {layout: (normalized, value) for layout, value in products.items()}
    outputs, checks = {}, {}
    for layout, out_sharding in layouts:
        label = f"{name}/collectives/continuation/{layout}"
        outputs[layout] = fixed_bank_outputs(rec, label, label, polar_continuation, banks, (rep, rep), out_sharding)
        compare_bitwise(rec, f"{label}/A_banks", outputs[layout]["direction_sharded"], outputs[layout]["replicated"])
        for bank_name, value in outputs[layout].items():
            for ref_layout, reference in references.items():
                key = f"{label}/{bank_name}/vs_reference_{ref_layout}"
                passed = compare_bitwise(rec, key, value, reference)
                checks[key] = dict(passed=passed, bitwise_equal=rec.report["comparisons"][key]["bitwise_equal"])
    for bank_name in banks:
        compare_bitwise(
            rec,
            f"{name}/collectives/{bank_name}/continuation_layouts",
            outputs["direction_sharded"][bank_name],
            outputs["replicated"][bank_name],
        )
    assert fingerprint(partials) == partial_hash and fingerprint(normalized) == normalized_hash
    rec.report.setdefault("frozen_partial_checks", {})[name] = dict(
        reference_checks=checks,
        original_path_interpretation_allowed=False,
        limitation=(
            "Materializing partials changes product compilation; explicit collectives may select different algorithms. "
            "Inspect compiled paths. Equal final endpoints do not prove internal identity or optimizer equivalence."
        ),
    )
    rec.persist()


def continuation_crossfeed(rec, name, products, traces, references, rep, sharding):
    """Hold X and each executable fixed while changing the supplied first Gram."""
    x = traces["replicated"]["normalized_input"]
    banks = {
        "replicated_A": (x, products["replicated"]["replicated"]),
        "contracting_A": (x, products["contracting_input"]["replicated"]),
    }
    rec.report.setdefault("continuation_banks", {})[name] = {
        key: dict(sha256=fingerprint(bank), x_sha256=fingerprint(bank[0]), A_sha256=fingerprint(bank[1]))
        for key, bank in banks.items()
    }
    outputs = {}
    for layout, out_sharding in [("replicated", rep), ("direction_sharded", sharding)]:
        label = f"{name}/continuation/{layout}"
        outputs[layout] = fixed_bank_outputs(rec, label, label, polar_continuation, banks, (rep, rep), out_sharding)
        rec.compare(label + "/A_banks", outputs[layout]["contracting_A"], outputs[layout]["replicated_A"])
        rec.report["comparisons"][label + "/A_banks"]["bitwise_equal"] = fingerprint(
            outputs[layout]["contracting_A"]
        ) == fingerprint(outputs[layout]["replicated_A"])
        for bank_name, result in outputs[layout].items():
            checks = {}
            for reference_layout in references:
                for kind, expected in (
                    ("uninstrumented", references[reference_layout]),
                    ("traced", traces[reference_layout]["final"]),
                ):
                    key = f"{label}/{bank_name}/vs_{kind}_{reference_layout}"
                    passed = rec.compare(key, result, expected)
                    exact = fingerprint(result) == fingerprint(expected)
                    rec.report["comparisons"][key]["bitwise_equal"] = exact
                    checks[f"{kind}_{reference_layout}"] = dict(passed=passed, bitwise_equal=exact)
            rec.report.setdefault("continuation_reference_checks", {})[f"{label}/{bank_name}"] = dict(
                checks=checks,
                same_layout_uninstrumented_passed=checks[f"uninstrumented_{layout}"]["passed"],
                original_path_interpretation_allowed=False,
                limitation="Separate continuation changes compilation; final agreement cannot verify internal identity",
            )
    for bank_name in banks:
        key = f"{name}/continuation/{bank_name}/layouts"
        rec.compare(key, outputs["direction_sharded"][bank_name], outputs["replicated"][bank_name])
    rec.persist()


def first_product_diagnostic(rec, common, rep, shardings):
    """Probe square q/k/v groups using the same refreshed-state recipe as attribution."""
    for name, state, sharding in zip(NAMES[:3], common[:3], shardings[:3], strict=True):
        powers = []
        for side, gram in [("left", state.left), ("right", state.right)]:
            normalized = rec.run(f"{name}/{side}/normalize", normalize, (gram,), (rep,), rep)
            vectors, values = rec.run(f"{name}/{side}/eigh", jax.lax.linalg.eigh, (normalized,), (rep,), (rep, rep))
            powers.append(rec.run(f"{name}/{side}/power", reconstruct, (vectors, values), (rep, rep), rep))
            del normalized, vectors, values
        inner = rec.run(f"{name}/inner", inner_product, (powers[0], state.momentum, powers[1]), (rep,) * 3, rep)
        del powers
        inner_sha = fingerprint(inner)
        if any(
            rec.report["config"].get(k) for k in ("normalization_control", "partition_control", "collective_control")
        ):
            if rec.report["config"].get("collective_control"):
                frozen_partial_collectives(rec, name, inner, rep, sharding)
            elif rec.report["config"].get("partition_control"):
                normalization_partition(rec, name, inner, rep, sharding)
            else:
                normalization_boundary(rec, name, inner, rep, sharding)
            assert fingerprint(inner) == inner_sha
            del inner
            jax.clear_caches()
            gc.collect()
            continue
        traces, instrumentation, exact_instrumentation, references = {}, {}, {}, {}
        for layout, out_sharding in [("replicated", rep), ("direction_sharded", sharding)]:
            reference = rec.run(f"{name}/identical_inner/{layout}", polar, (inner,), (rep,), out_sharding)
            trace_shardings = {
                key: rep if key == "norm" else out_sharding
                for key in ("norm", "normalized_input", "A", "C", "B", "D", "X1", "final")
            }
            traces[layout] = rec.run(f"{name}/products/{layout}", polar_products, (inner,), (rep,), trace_shardings)
            gate = f"{name}/products/{layout}/final_vs_uninstrumented"
            instrumentation[layout] = rec.compare(gate, traces[layout]["final"], reference)
            exact_instrumentation[layout] = fingerprint(traces[layout]["final"]) == fingerprint(reference)
            rec.report["comparisons"][gate]["bitwise_equal"] = exact_instrumentation[layout]
            if rec.report["config"].get("continuation_control"):
                references[layout] = reference
            del reference
        first_nonzero, first_failure = None, None
        for key in ("norm", "normalized_input", "A", "C", "B", "D", "X1", "final"):
            left, right = traces["direction_sharded"][key], traces["replicated"][key]
            passed = rec.compare(f"{name}/products/layouts/{key}", left, right)
            equal = fingerprint(left) == fingerprint(right)
            rec.report["comparisons"][f"{name}/products/layouts/{key}"]["bitwise_equal"] = equal
            if first_nonzero is None and not equal:
                first_nonzero = key
            if first_failure is None and not passed:
                first_failure = key
        # CPU-equal controls still exercise the fixed-executable harness at A.
        selected = first_nonzero if first_nonzero in ("A", "C", "B", "D", "X1") else None
        if first_nonzero is None or rec.report["config"].get("contracting_control"):
            selected = "A"
        rec.report.setdefault("product_decisions", {})[name] = dict(
            inner_sha256=inner_sha,
            instrumentation_passed=instrumentation,
            instrumentation_bitwise_equal=exact_instrumentation,
            first_nonzero=first_nonzero,
            first_tolerance_failure=first_failure,
            original_path_interpretation_allowed=all(exact_instrumentation.values()),
            selected_crossfeed=selected,
            crossfeed_scope="Instrumented-path perturbation control, not proof of original internal values",
            selection_reason=(
                "Preselected A contraction-split control"
                if rec.report["config"].get("contracting_control")
                else "First nonzero first-iteration stage, or A harness control when all traced leaves match"
            ),
        )
        rec.persist()
        if selected is not None:
            products = product_crossfeed(rec, name, selected, traces, rep, sharding)
            if rec.report["config"].get("continuation_control"):
                continuation_crossfeed(rec, name, products, traces, references, rep, sharding)
            del products
        assert fingerprint(inner) == inner_sha
        del traces, inner, references
        jax.clear_caches()
        gc.collect()


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
            or "/products/" in name
            or "/normalization/" in name
            or "/partition/" in name
            or "/collectives/" in name
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


def fixed_bank_outputs(rec, label, group, fn, banks, state_shardings, output_shardings):
    """Compile once, evaluate each input bank, then replay the first bank."""
    first = next(iter(banks))
    replay = first + "_replay"
    order = [*banks, replay]
    input_hashes = {name: fingerprint(bank) for name, bank in banks.items()}
    rec.report["active_stage"] = group + "/compile"
    rec.persist()
    exemplar = jax.tree.map(
        lambda x, sh: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sh), banks[first], state_shardings
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
        invocation_order=order,
        input_sha256=input_hashes,
    )
    rec.persist()
    outputs = {}
    for index, bank_name in enumerate(order):
        bank = banks[first if bank_name == replay else bank_name]
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
    rec.compare(group + "/replay", outputs[replay], outputs[first])
    rec.report["fixed_executables"][label]["replay_bitwise_equal"] = fingerprint(outputs.pop(replay)) == fingerprint(
        outputs[first]
    )
    assert {name: fingerprint(bank) for name, bank in banks.items()} == input_hashes
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
        outputs = fixed_bank_outputs(
            rec,
            label,
            f"fixed_direction/{label}",
            partial(directions, alpha=0.25, eigh_fn=eig_fn),
            banks,
            state_shardings,
            output_shardings,
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


def full_input_capture(rec, inputs, matrix_eigh, input_shardings, output_shardings, replicated):
    """Compare untouched full updates, unchanged traced programs and extra outputs."""
    input_sha = fingerprint(inputs)
    captures, references = {}, {}
    rec.report["capture_inputs_sha256"] = input_sha
    rec.report["capture_programs"] = {}
    rec.report["capture_gates"] = {}
    for label, solver in [("original", jax.lax.linalg.eigh), ("matrix", matrix_eigh)]:
        fn = full_update if label == "original" else partial(full_update, eigh_fn=solver)
        reference = rec.run("fused_" + label, fn, inputs, input_shardings, output_shardings)
        references[label] = reference
        exemplars = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), inputs)
        evaluate, capture, metadata, source = capture_program(fn, exemplars)
        payload = gzip.compress(source.encode(), mtime=0)
        filename = "capture_program_" + label + ".jaxpr.txt.gz"
        (rec.local / filename).write_bytes(payload)
        (rec.destination / filename).write_bytes(payload, requests_timeout=REPORT_REQUEST_TIMEOUT)
        metadata["jaxpr_artifact"] = dict(
            uri=str(rec.destination / filename),
            sha256=hashlib.sha256(payload).hexdigest(),
            compressed_bytes=len(payload),
        )
        rec.report["capture_programs"][label] = metadata
        rec.persist()
        evaluated = fixed_bank_outputs(
            rec,
            label + "_evaluate",
            "capture/" + label + "/evaluate",
            evaluate,
            {"identical_inputs": inputs},
            input_shardings,
            output_shardings,
        )["identical_inputs"]
        capture_shardings = (output_shardings, tuple((replicated, replicated) for _ in NAMES[:3]))
        captured = fixed_bank_outputs(
            rec,
            label + "_capture",
            "capture/" + label + "/extra_outputs",
            capture,
            {"identical_inputs": inputs},
            input_shardings,
            capture_shardings,
        )["identical_inputs"]
        gates = {}
        for kind, actual, expected in [
            ("evaluate_vs_untouched", evaluated, reference),
            ("capture_vs_untouched", captured[0], reference),
            ("capture_vs_evaluate", captured[0], evaluated),
        ]:
            key = "capture/" + label + "/" + kind
            gates[kind] = rec.compare(key, actual, expected)
            rec.report["comparisons"][key]["bitwise_equal"] = fingerprint(actual) == fingerprint(expected)
        rec.report["capture_gates"][label] = dict(
            comparisons_passed=gates,
            reference_agreement=all(gates.values()),
            original_internal_identity_established=False,
        )
        captures[label] = captured
        del evaluated, reference, evaluate, capture, source
        jax.clear_caches()
        gc.collect()
    passed = rec.compare("fused_candidate_vs_original", references["matrix"], references["original"])
    rec.compare("capture/candidate_vs_original", captures["matrix"][0], captures["original"][0])
    for index, group in enumerate(NAMES[:3]):
        for field, name in enumerate(("inner", "norm_plus_eps")):
            actual, expected = captures["matrix"][1][index][field], captures["original"][1][index][field]
            key = "capture/values/" + group + "/" + name
            rec.compare(key, actual, expected)
            rec.report["comparisons"][key]["bitwise_equal"] = fingerprint(actual) == fingerprint(expected)
            rec.report["comparisons"][key]["input_reference_gates_passed"] = all(
                g["reference_agreement"] for g in rec.report["capture_gates"].values()
            )
    assert fingerprint(inputs) == input_sha
    assert len(rec.report["stages"]) == 12 and len(rec.report["comparisons"]) == 18
    rec.persist()
    return passed


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
    mode.add_argument("--polar-products", action="store_true")
    mode.add_argument("--polar-contracting", action="store_true")
    mode.add_argument("--polar-continuation", action="store_true")
    mode.add_argument("--polar-normalization", action="store_true")
    mode.add_argument("--polar-partition", action="store_true")
    mode.add_argument("--polar-collectives", action="store_true")
    mode.add_argument("--full-input-capture", action="store_true")
    args = parser.parse_args()
    product_mode = (
        args.polar_products
        or args.polar_contracting
        or args.polar_continuation
        or args.polar_normalization
        or args.polar_partition
        or args.polar_collectives
    )
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
        polar_compiler=args.polar_compiler or args.state_crossfeed or product_mode or args.full_input_capture,
        full_input_capture=args.full_input_capture,
        capture_helper_sha256=hashlib.sha256(
            Path(__file__).with_name("optimizer_input_capture.py").read_bytes()
        ).hexdigest(),
        state_crossfeed=args.state_crossfeed,
        polar_products=product_mode,
        contracting_control=args.polar_contracting or args.polar_continuation,
        continuation_control=args.polar_continuation,
        normalization_control=args.polar_normalization,
        partition_control=args.polar_partition,
        collective_control=args.polar_collectives,
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
    if args.full_input_capture:
        inputs = jax.device_get((gradients, states))
        del gradients, states
        jax.clear_caches()
        gc.collect()
        fused_pass = full_input_capture(rec, inputs, matrix_eigh, (gs, ss), (gs, ss), rep)
        finish(rec, tracker, fused_pass, args.smoke)
        return
    common = rec.run("common_refresh", refresh, (gradients, states), (gs, ss), ss)
    if product_mode:
        del gradients, states
        jax.clear_caches()
        gc.collect()
        first_product_diagnostic(rec, common, rep, gs)
        finish(rec, tracker, None, args.smoke)
        return
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
