# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline correctness checks; CPU timings are never performance evidence."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import export
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.optim.ema_gram import init_gram_state, relative_gram_power
from optimizer_parallel import eig_parallel_update, local_update, parallel_update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layout", choices=("whole-update", "eigh-only"), default="whole-update")
    parser.add_argument("--input-layout", choices=("replicated", "embedding-sharded"), default="replicated")
    args = parser.parse_args()
    assert jax.default_backend() == "cpu" and len(jax.devices()) == 16
    if args.input_layout == "embedding-sharded":
        mesh = Mesh(np.array(jax.devices()).reshape(16, 1), ("data", "model"))
        axis_name = "data"
        shapes = [(11, 32, 32), (11, 128, 32), (11, 32, 128), (32, 128)]
    else:
        mesh = Mesh(np.array(jax.devices()).reshape(1, 1, 16), ("data", "model", "matrix"))
        axis_name = "matrix"
        shapes = [(11, 8, 8), (11, 32, 8), (11, 8, 32), (8, 32)]
    replicated = NamedSharding(mesh, P())
    rng = np.random.default_rng(20)
    candidate_fn = parallel_update if args.layout == "whole-update" else eig_parallel_update
    checks = []
    for shape in shapes:
        gradient_spec = P(*([None] * (len(shape) - 1)), "data")
        input_sharding = NamedSharding(mesh, gradient_spec) if args.input_layout == "embedding-sharded" else replicated
        for alpha in [0.0, 0.25, 0.5]:
            for ridge in [1e-4, 1e-2]:
                kwargs = dict(alpha=alpha, ridge=ridge, eigh_implementation=None)
                serial = jax.jit(partial(local_update, **kwargs), out_shardings=replicated)
                parallel = jax.jit(
                    partial(candidate_fn, mesh=mesh, axis_name=axis_name, **kwargs),
                    out_shardings=replicated,
                )
                gradient = jax.device_put(np.zeros(shape, np.float32), input_sharding)
                reference_state = init_gram_state(gradient)
                candidate_state = reference_state
                maximum_error = 0.0
                failures = []
                for step in range(3):
                    if step:
                        gradient = jax.device_put(rng.normal(size=shape).astype(np.float32) * 0.01, input_sharding)
                    expected, reference_state = jax.block_until_ready(serial(gradient, reference_state))
                    actual, candidate_state = jax.block_until_ready(parallel(gradient, candidate_state))
                    for leaf_index, (a, b) in enumerate(
                        zip(
                            jax.tree.leaves((actual, candidate_state)),
                            jax.tree.leaves((expected, reference_state)),
                            strict=True,
                        )
                    ):
                        a, b = np.asarray(a), np.asarray(b)
                        assert np.isfinite(a).all()
                        if not np.allclose(a, b, rtol=1e-5, atol=1e-5):
                            failures.append(
                                dict(
                                    step=step,
                                    leaf=leaf_index,
                                    maximum_absolute_error=float(np.max(np.abs(a - b))),
                                    relative_frobenius_error=float(
                                        np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
                                    ),
                                )
                            )
                        maximum_error = max(maximum_error, float(np.max(np.abs(a - b))))
                checks.append(
                    dict(
                        shape=shape,
                        alpha=alpha,
                        ridge=ridge,
                        steps=3,
                        maximum_absolute_error=maximum_error,
                        failures=failures,
                        passed=not failures,
                    )
                )
    # The public lax API and the previous numpy API must agree before choosing
    # any different TPU solver. This is separate from solver validation.
    power_checks = []
    for n in [8, 32]:
        x = rng.normal(size=(3, n, max(1, n // 2))).astype(np.float32)
        grams = x @ x.swapaxes(-1, -2)
        for alpha in [0.125, 0.25, 1 / 3, 0.5]:
            for ridge in [1e-4, 1e-2]:
                gram = jnp.asarray(grams)
                norm = gram / (jnp.trace(gram, axis1=-2, axis2=-1) / n)[..., None, None]
                vals, vecs = jnp.linalg.eigh(norm)
                expected = (vecs * (jnp.maximum(vals, 0) + ridge)[..., None, :] ** -alpha) @ vecs.swapaxes(-1, -2)
                actual = relative_gram_power(gram, alpha, ridge)
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
                power_checks.append(dict(dimension=n, alpha=alpha, ridge=ridge))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lowerings = []
    for n in [1024, 4096]:
        for name, solver in [("default", None), ("jacobi", jax.lax.linalg.EighImplementation.JACOBI)]:
            fn = jax.jit(partial(relative_gram_power, alpha=0.25, ridge=1e-4, eigh_implementation=solver))
            exported = export.export(fn, platforms=("tpu",))(jax.ShapeDtypeStruct((11, n, n), np.float32))
            hlo = exported.mlir_module()
            path = output.parent / f"power-{name}-{n}.stablehlo.txt"
            path.write_text(hlo)
            lowerings.append(
                dict(
                    solver=name,
                    shape=[11, n, n],
                    file=str(path),
                    sha256=hashlib.sha256(hlo.encode()).hexdigest(),
                    stablehlo_while_count=hlo.count("stablehlo.while"),
                    eigh_custom_call_count=hlo.count("custom_call @Eigh"),
                )
            )
    report = dict(
        jax_version=jax.__version__,
        layout=args.layout,
        input_layout=args.input_layout,
        backend=jax.default_backend(),
        devices=len(jax.devices()),
        layer_parallel_checks=checks,
        numpy_to_lax_checks=power_checks,
        tpu_lowerings=lowerings,
        tpu_execution_validated=False,
        all_parallel_checks_passed=all(c["passed"] for c in checks),
        tolerance=dict(rtol=1e-5, atol=1e-5),
        limitations="CPU comparisons test matrix assignment/padding only. TPU Jacobi numerical accuracy, "
        "runtime, memory and training MFU remain unmeasured.",
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                passed=report["all_parallel_checks_passed"],
                parallel_cases=len(checks),
                failed_cases=sum(not c["passed"] for c in checks),
                power_cases=len(power_checks),
                output=str(output),
            )
        )
    )
    assert report["all_parallel_checks_passed"], "Candidate has not passed the unchanged numerical gate; see report"


if __name__ == "__main__":
    main()
