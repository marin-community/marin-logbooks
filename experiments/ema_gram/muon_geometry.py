# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare six fixed Muon directions with calibration locked before evaluation."""

import hashlib
import json
import time

import equinox as eqx
import jax
import numpy as np
import optax

from experiments.ema_gram.geometry_math import quadratic_step, refinement_lengths
from experiments.ema_gram.gradient_curvature import PathMeasurement, measurement_functions
from experiments.ema_gram.gradient_curvature_math import tree_add, tree_dot, tree_norm, tree_scale, weighted_gradient
from experiments.ema_gram.hessian import host_array
from experiments.ema_gram.muon_geometry_kernels import VARIANTS
from experiments.ema_gram.muon_geometry_updates import (
    add_f5_feedback,
    momentum_state,
    native_muon_functions,
    tangent_functions,
)
from experiments.ema_gram.power_prediction import mean_product


def telemetry_records(metrics):
    """Retain failed-solve flags without serializing nonfinite JSON numbers."""
    records = []
    for bank, values in enumerate(metrics):
        arrays = {k: np.asarray(host_array(v)) for k, v in values.items()}
        for index in range(len(next(iter(arrays.values())))):
            row = dict(bank=bank, matrix=index)
            for key, array in arrays.items():
                value = array[index].item()
                row[key] = value if np.isfinite(value) else None
            records.append(row)
    return records


def run_muon_geometry(
    *, state, model, native_model, cfg, panels, weights, args, protocol, save, progress, manifest_hash, save_arrays
):
    del save_arrays
    value_grad, hvp, loss_fn = measurement_functions(cfg.trainer.compute_axis_mapping)
    native_grad, updates, reference_config = native_muon_functions(cfg, protocol["reference_learning_rate"])
    tangent_fn, finite_difference = tangent_functions(cfg)
    definition = jax.tree.structure(model)
    calibration, cal_weights = panels["calibration"], weights["calibration"]
    reference = PathMeasurement(model, calibration, cal_weights, [len(calibration)], value_grad, hvp, loss_fn, progress)
    baseline, gradient = reference.baseline[len(calibration)]
    output = dict(
        state=args.state,
        counter=int(state.step),
        data_manifest_sha256=manifest_hash,
        reference_learning_rate=reference_config.learning_rate,
        reference_adam_learning_rate=reference_config.adam_lr,
        scale_scope=protocol["reference_step_scope"],
        calibration_baseline=baseline,
        calibration_weights=cal_weights,
        evaluation_weights=weights["evaluation"],
        rows=[],
        derivative_checks=[],
        choices=[],
    )
    # Keep one exact copy per direction in host RAM; never regenerate after calibration.
    directions, native_norms = {}, []
    microbatches = protocol["native_update_sequences"] // protocol["measurement_microbatch_sequences"]
    for draw in range(protocol["panels"]["update_draws"]):
        start = draw * microbatches
        batches = panels["update"][start : start + microbatches]
        draw_weights = weights["update"][start : start + microbatches]
        gradient_started = time.perf_counter()
        losses, grad = weighted_gradient(native_model, batches, draw_weights, native_grad)
        gradient_seconds = time.perf_counter() - gradient_started
        plain_started = time.perf_counter()
        plain, post = updates["muon"](grad, state.opt_state, eqx.filter(native_model, eqx.is_inexact_array))
        jax.block_until_ready(plain)
        plain_seconds = time.perf_counter() - plain_started
        tangent_started = time.perf_counter()
        tangent, metrics = tangent_fn(post, grad)
        jax.block_until_ready(tangent)
        tangent_seconds = time.perf_counter() - tangent_started
        checks = []
        for epsilon in protocol["f5_fd_relative_steps"]:
            fd = finite_difference(post, grad, epsilon)
            error = tree_norm(tree_add(fd, tangent, -1)) / max(tree_norm(tangent), 1e-30)
            checks.append(dict(epsilon=epsilon, relative_error=error if np.isfinite(error) else None))
            if len(checks) >= 2 and all(
                x["relative_error"] is not None and x["relative_error"] < 0.02 for x in checks[-2:]
            ):
                break
        passed = len(checks) >= 2 and all(
            x["relative_error"] is not None and x["relative_error"] < 0.02 for x in checks[-2:]
        )
        diagnostic = dict(draw=draw, F5_finite_difference=checks, passed=passed, matrices=telemetry_records(metrics))
        output["derivative_checks"].append(diagnostic)
        save("muon-partial.json", output)
        if not passed:
            raise ValueError("Pinned F5 tangent failed two-scale finite differences")
        for variant in VARIANTS:
            update_started = time.perf_counter()
            if variant == "muon":
                update = plain
            elif variant == "F5_tangent_0.05":
                update = add_f5_feedback(plain, tangent, protocol["reference_learning_rate"])
            else:
                update, advanced = updates[variant](
                    grad, state.opt_state, eqx.filter(native_model, eqx.is_inexact_array)
                )
                if tree_norm(tree_add(momentum_state(post), momentum_state(advanced), -1)) != 0:
                    raise ValueError("Muon variants advanced different momentum")
            norm = float(optax.global_norm(update))
            update_seconds = time.perf_counter() - update_started
            if variant == "muon":
                update_seconds += plain_seconds
            elif variant == "F5_tangent_0.05":
                update_seconds += plain_seconds + tangent_seconds
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("Invalid complete Muon update")
            direction = tree_scale(definition.unflatten(jax.tree.leaves(update)), -1 / norm)
            arrays = tuple(np.array(host_array(x), copy=True) for x in jax.tree.leaves(direction))
            digest = hashlib.sha256()
            for array in arrays:
                digest.update(array.tobytes())
            directions[draw, variant] = arrays
            if variant == "muon":
                native_norms.append(norm)
            output["rows"].append(
                dict(
                    draw=draw,
                    variant=variant,
                    reference_update_norm=norm,
                    direction_sha256=digest.hexdigest(),
                    update_loss=float(np.average(losses, weights=draw_weights)),
                    update_gradient_norm=tree_norm(grad),
                    gradient_seconds=gradient_seconds,
                    native_seconds=update_seconds,
                    timing_scope="Wall time including first-call compilation; tangent includes correction telemetry",
                    matrix_gradient_clipping="None in pinned native optimizer; Adam branch retains its own clipping",
                )
            )
            del update, direction
        progress(dict(event="muon_directions_frozen", draw=draw, directions=len(directions)))
        del grad, plain, post, tangent, metrics, fd
    radius_reference = float(np.median(native_norms))
    radius = 4 * radius_reference
    grid = [x * radius_reference for x in protocol["length_grid"]]
    output.update(reference_norm=radius_reference, radius=radius)

    def restore_direction(row):
        leaves = [
            jax.device_put(array, target.sharding)
            for array, target in zip(directions[row["draw"], row["variant"]], jax.tree.leaves(model), strict=True)
        ]
        return definition.unflatten(leaves)

    first_direction, first_product = None, None
    for row in output["rows"]:
        direction = restore_direction(row)
        product = mean_product(model, direction, calibration, cal_weights, hvp)
        alignment, curvature = float(tree_dot(gradient, direction)), float(tree_dot(direction, product))
        if row["draw"] == 0 and row["variant"] == "muon":
            check = reference.derivative_check(direction, 0)
            output["HVP_validation"] = check
            save("muon-partial.json", output)
            if not check["passed"]:
                raise ValueError("Muon reference HVP validation failed")
            first_direction, first_product = direction, product
        if row["draw"] == 0 and row["variant"] == "blend_0.02":
            symmetry = abs(float(tree_dot(first_direction, product) - tree_dot(direction, first_product)))
            symmetry /= max(tree_norm(product), tree_norm(first_product), 1e-30)
            output["HVP_symmetry_relative_error"] = symmetry
            if symmetry > 1e-4:
                save("muon-partial.json", output)
                raise ValueError("Muon reference HVP symmetry failed")
            first_direction, first_product = None, None
        step, predicted = quadratic_step(alignment, curvature, radius)
        cache = {0.0: baseline}
        for length in grid:
            if length not in cache:
                candidate = tree_add(model, direction, -length)
                cache[length] = [float(loss_fn(candidate, b)) for b in calibration]
        refined, boundary = refinement_lengths(grid, [np.average(cache[s], weights=cal_weights) for s in grid])
        for length in refined:
            cache[length] = [float(loss_fn(tree_add(model, direction, -length), b)) for b in calibration]
        chosen = min(cache, key=lambda s: (np.average(cache[s], weights=cal_weights), s))
        row.update(
            alignment=alignment,
            curvature=curvature,
            H_product_norm=tree_norm(product),
            quadratic_step=step,
            predicted_decrease=predicted,
            direct_step=chosen,
            grid_boundary_winner=boundary,
            direct_at_radius=bool(chosen == radius),
            direct_at_zero=bool(chosen == 0),
            calibration=[
                dict(length=s, losses=v, mean=float(np.average(v, weights=cal_weights)))
                for s, v in sorted(cache.items())
            ],
        )
        save("muon-partial.json", output)
        progress(dict(event="muon_calibration", draw=row["draw"], variant=row["variant"], direct_step=chosen))
        del direction, product
    for draw in range(protocol["panels"]["update_draws"]):
        rows = [r for r in output["rows"] if r["draw"] == draw]
        output["choices"].append(
            dict(
                draw=draw,
                quadratic=max(rows, key=lambda r: r["predicted_decrease"])["variant"],
                alignment=max(rows, key=lambda r: r["alignment"])["variant"],
                calibration_best=min(rows, key=lambda r: min(x["mean"] for x in r["calibration"]))["variant"],
            )
        )
    locked = json.loads(json.dumps(output))
    save("locked-predictions.json", locked)
    output["locked_predictions_sha256"] = hashlib.sha256(json.dumps(locked, sort_keys=True).encode()).hexdigest()
    progress(dict(event="predictions_locked", rows=len(output["rows"]), sha256=output["locked_predictions_sha256"]))
    evaluation = panels["evaluation"]
    eval_baseline = [float(loss_fn(model, b)) for b in evaluation]
    output["evaluation_baseline"] = eval_baseline
    for row in output["rows"]:
        direction = restore_direction(row)
        cache = {0.0: eval_baseline}
        row["evaluation"] = {}
        lengths = [
            ("reference_native", row["reference_update_norm"]),
            ("equal", radius_reference),
            ("direct", row["direct_step"]),
            ("quadratic", row["quadratic_step"]),
        ] + [(f"equal_{x:g}", x * radius_reference) for x in protocol["length_grid"]]
        for name, length in lengths:
            candidate = tree_add(model, direction, -length)
            if length not in cache:
                cache[length] = [float(loss_fn(candidate, b)) for b in evaluation]
            row["evaluation"][name] = dict(
                length=length,
                losses=cache[length],
                loss_change=float(np.average(np.asarray(cache[length]) - eval_baseline, weights=weights["evaluation"])),
                predicted_change=-length * row["alignment"] + 0.5 * length**2 * row["curvature"],
                actual_norm=tree_norm(tree_add(candidate, model, -1)),
            )
        save("muon-results.json", output)
        progress(
            dict(
                event="muon_heldout",
                draw=row["draw"],
                variant=row["variant"],
                loss_change=row["evaluation"]["direct"]["loss_change"],
            )
        )
        del directions[row["draw"], row["variant"]]
    return output
