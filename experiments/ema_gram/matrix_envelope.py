# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Block Hessian responses with disjoint calibration and reserved test subspaces."""

import hashlib
import json
import time

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from levanter.optim.util import flatten_linear_layers

from experiments.ema_gram.envelope_path import load_path_baseline, measure_path
from experiments.ema_gram.geometry_math import envelope_components, orthonormal_matrices, rotated_gradient
from experiments.ema_gram.gradient_curvature import production_functions, production_updates
from experiments.ema_gram.hessian import FD_STEPS, block_functions, host_array


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def probe_bases(gradient, optimizer_directions, seed):
    rng = np.random.default_rng(seed)
    u, _, vh = np.linalg.svd(np.asarray(gradient, np.float64), full_matrices=False)
    calibration = [(f"gradient_singular_{i}", np.outer(u[:, i], vh[i])) for i in range(4)]
    calibration += [(f"random_calibration_{i}", rng.normal(size=gradient.shape)) for i in range(4)]
    testing = [(f"optimizer_{alpha}", direction) for alpha, direction in optimizer_directions.items()]
    testing += [(f"gradient_singular_{i}", np.outer(u[:, i], vh[i])) for i in (4, 5)]
    testing += [(f"random_test_{i}", rng.normal(size=gradient.shape)) for i in range(8)]
    cal, cal_names = orthonormal_matrices(calibration, 8)
    test, test_names = orthonormal_matrices(testing, 8)
    return cal, test, dict(calibration=cal_names, test=test_names)


def run_matrix_envelope(
    *, state, model, native_model, cfg, panels, weights, args, protocol, save, progress, manifest_hash, save_arrays
):
    if "middle" not in args.state:
        raise ValueError("Initial matrix-envelope study uses only the two middle states")
    plan = json.loads(args.plan.read_text()) if args.plan else None
    if args.stage in ("test", "path") and plan is None:
        raise ValueError("Test responses require published frozen envelope coefficients")
    if args.stage == "path" and args.path_inputs is None:
        raise ValueError("Path extension requires validated baseline test response artifacts")
    if plan is not None:
        save("frozen-envelope-plan.json", plan)
    reference, reference_weights = panels["calibration"], weights["calibration"]
    normalized = np.asarray(reference_weights) / np.sum(reference_weights)
    native_grad, updates = production_functions(cfg)
    _, update_gradient = native_grad(native_model, panels["update"][0])
    native_blocks = {}
    for alpha in (0.125, 0.25):
        update, _ = production_updates(state, native_model, update_gradient, updates["full", alpha], False)
        native_blocks[alpha] = {
            jax.tree_util.keystr(path): host_array(value).reshape((-1, *value.shape[-2:]))[0]
            for path, value in jax.tree_util.tree_flatten_with_path(flatten_linear_layers(update))[0]
            if any(name in jax.tree_util.keystr(path) for name in ("self_attn.q_proj.weight", "mlp.gate_proj.weight"))
        }
    del update, update_gradient
    blocks = [
        (jax.tree_util.keystr(path), value)
        for path, value in jax.tree_util.tree_flatten_with_path(flatten_linear_layers(model))[0]
        if any(name in jax.tree_util.keystr(path) for name in ("self_attn.q_proj.weight", "mlp.gate_proj.weight"))
    ]
    if len(blocks) != 2:
        raise ValueError("Expected exactly attention query and MLP gate matrix banks")
    output = dict(
        state=args.state,
        stage=args.stage,
        counter=int(state.step),
        data_manifest_sha256=manifest_hash,
        reference_weights=reference_weights,
        rows=[],
    )
    state_index = next(i for i, s in enumerate(protocol["states"]) if s["state"] == args.state)
    for block_index, (path, bank) in enumerate(blocks):
        started = time.perf_counter()
        label = "attention" if "q_proj" in path else "mlp"
        if args.stage == "path" and not plan["path_gate"][label]["eligible"]:
            progress(dict(event="path_ineligible", block=label, gate=plan["path_gate"][label]))
            continue
        matrix = bank.reshape((-1, *bank.shape[-2:]))[0]
        loss, grad, hvp = block_functions(model, path, 1e-7)

        def compile_fp32(function):
            def wrapped(*inputs):
                with jax.default_matmul_precision("highest"):
                    return function(*inputs)

            return hax.named_jit(wrapped, axis_resources=cfg.trainer.compute_axis_mapping)

        loss, grad, hvp = [compile_fp32(f) for f in (loss, grad, hvp)]

        def mean_gradient(point, matrix=matrix, grad=grad):
            total = jnp.zeros_like(matrix)
            for b, w in zip(reference, normalized, strict=True):
                total = total + w * grad(point, model, b)
            return host_array(total).astype(np.float64)

        def mean_hvp(point, direction, matrix=matrix, hvp=hvp):
            total = jnp.zeros_like(matrix)
            for b, w in zip(reference, normalized, strict=True):
                total = total + w * hvp(point, jnp.asarray(direction), model, b)
            return host_array(total).astype(np.float64)

        gradient = mean_gradient(matrix)
        optimizer_directions = {a: -v[path] / np.linalg.norm(v[path]) for a, v in native_blocks.items()}
        seed = protocol["seeds"]["probe"] + 1000 * state_index + block_index
        cal, test, names = probe_bases(gradient, optimizer_directions, seed)
        identities = dict(
            gradient_sha256=array_hash(gradient),
            calibration_basis_sha256=array_hash(cal),
            test_basis_sha256=array_hash(test),
            data_manifest_sha256=manifest_hash,
        )
        if plan is not None and identities != plan["inputs"][args.state][label]:
            raise ValueError("Frozen coefficient plan and regenerated block/data differ")
        if args.stage == "path":
            source, arrays = load_path_baseline(args, plan, label, identities)
            if not np.array_equal(arrays["basis"], test) or not np.array_equal(arrays["gradient"], gradient):
                raise ValueError("Path baseline arrays differ from regenerated starting state")
            result = measure_path(
                matrix=matrix,
                gradient=gradient,
                basis=test,
                baseline_responses=arrays["responses"],
                directions=optimizer_directions,
                length=float(np.linalg.norm(native_blocks[0.125][path])),
                mean_gradient=mean_gradient,
                mean_hvp=mean_hvp,
                losses=lambda point, loss=loss: [float(loss(point, model, batch)) for batch in reference],
                weights=reference_weights,
                fits=plan["fits"][label],
                rotation_seed=protocol["seeds"]["rotation"] + 1000 * state_index + block_index,
                save_arrays=save_arrays,
                progress=progress,
                label=label,
            )
            result["baseline_source"] = source
            output["rows"].append(result)
            save("envelope-path-results.json", output)
            continue
        basis = cal if args.stage == "screen" else test
        responses = []
        for index, direction in enumerate(basis):
            responses.append(mean_hvp(matrix, direction))
            progress(dict(event="block_hvp", block=label, stage=args.stage, direction=index))
        responses = np.stack(responses)
        projected = basis.reshape(8, -1).astype(np.float64) @ responses.reshape(8, -1).T
        symmetry = float(np.linalg.norm(projected - projected.T) / max(np.linalg.norm(projected), 1e-30))
        checks = []
        for index in (0, 7):
            previous = False
            trials = []
            for delta in FD_STEPS:
                fd = (
                    mean_gradient(matrix + delta * jnp.asarray(basis[index]))
                    - mean_gradient(matrix - delta * jnp.asarray(basis[index]))
                ) / (2 * delta)
                error = float(np.linalg.norm(fd - responses[index]) / max(np.linalg.norm(responses[index]), 1e-30))
                trials.append(
                    dict(delta=delta, relative_error=error, absolute_error=float(np.linalg.norm(fd - responses[index])))
                )
                progress(dict(event="block_fd", block=label, direction=index, delta=delta, relative_error=error))
                passed = error < 0.02
                if passed and previous:
                    break
                previous = passed
            checks.append(
                dict(
                    direction=index,
                    passed=bool(len(trials) > 1 and all(t["relative_error"] < 0.02 for t in trials[-2:])),
                    trials=trials,
                )
            )
        row = dict(
            block=label,
            path=path,
            layer_index=0,
            shape=list(matrix.shape),
            identities=identities,
            basis_names=names,
            projected_hessian=projected.tolist(),
            symmetry_relative=symmetry,
            derivative_checks=checks,
            gradient_norm=float(np.linalg.norm(gradient)),
            basis_gram=(basis.reshape(8, -1).astype(np.float64) @ basis.reshape(8, -1).T).tolist(),
            native_block_norms={str(a): float(np.linalg.norm(v[path])) for a, v in native_blocks.items()},
            optimizer_coordinates={
                str(a): (basis.reshape(8, -1) @ e.ravel()).tolist() for a, e in optimizer_directions.items()
            },
            native_seconds=time.perf_counter() - started,
        )
        rotated = rotated_gradient(gradient, protocol["seeds"]["rotation"] + 1000 * state_index + block_index)
        row["envelope_components"] = envelope_components(gradient, basis).tolist()
        row["rotated_components"] = envelope_components(rotated, basis).tolist()
        row["rotation"] = dict(
            method="Independent DCT/sign/permutation orthogonal transforms",
            seed=protocol["seeds"]["rotation"] + 1000 * state_index + block_index,
            gradient_norm_relative_error=abs(np.linalg.norm(rotated) / np.linalg.norm(gradient) - 1),
        )
        row["arrays"] = save_arrays(
            f"{label}-{args.stage}.npz",
            gradient=gradient,
            basis=basis,
            calibration_basis=cal,
            test_basis=test,
            responses=responses,
            parameter=host_array(matrix),
            **{f"optimizer_{a}": v[path] for a, v in native_blocks.items()},
        )
        output["rows"].append(row)
        save("envelope-results.json", output)
        if symmetry >= 1e-4 or not all(x["passed"] for x in checks):
            raise ValueError("Block HVP accuracy or symmetry validation failed")
        del gradient, responses, cal, test, rotated
    return output
