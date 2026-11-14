# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Separate saved momentum and Gram histories with disjoint fixed-weight replicas."""

import hashlib
import json
import time

import equinox as eqx
import jax
import numpy as np
import optax

from experiments.ema_gram.geometry_math import quadratic_step, refinement_lengths
from experiments.ema_gram.gradient_curvature import PathMeasurement, measurement_functions, production_functions
from experiments.ema_gram.gradient_curvature_math import tree_add, tree_dot, tree_norm, tree_scale
from experiments.ema_gram.hessian import host_array
from experiments.ema_gram.history_geometry import (
    collect_frozen_history,
    combine_histories,
    pool_raw_histories,
    replace_matrix_history,
)
from experiments.ema_gram.power_prediction import host_inner_product, host_vector_norm, mean_product
from experiments.ema_gram.probe import matrix_state
from experiments.ema_gram.state_fingerprint import describe_arrays


def run_history_comparison(
    *, state, model, native_model, cfg, panels, weights, args, protocol, save, progress, manifest_hash, save_arrays
):
    recipe = protocol["history"]
    native_grad, functions = production_functions(cfg, alphas=(0.125, 0.25), modes=("full",))
    value_grad, hvp, loss_fn = measurement_functions(cfg.trainer.compute_axis_mapping)
    saved = matrix_state(native_model, state.opt_state).optimizer
    seen = set(json.loads(args.exclusions.read_text()))
    for batches in panels.values():
        for batch in batches:
            seen.update(hashlib.sha256(x.tobytes()).hexdigest() for x in host_array(batch.tokens.array))
    fresh, records = [], []
    for replica, seed in enumerate(recipe["seeds"]):
        history, seen, record = collect_frozen_history(
            cfg=cfg,
            native_model=native_model,
            opt_state=state.opt_state,
            gradient_function=native_grad,
            seed=seed,
            count=recipe["batches_per_replica"],
            excluded_hashes=seen,
            save=save,
            save_arrays=save_arrays,
            progress=progress,
            replica=replica,
        )
        fresh.append(history)
        records.append(record)
    pools = {"replica0": fresh[0], "replica1": fresh[1], "pooled": pool_raw_histories(fresh)}
    conditions = [dict(name="saved", replica="saved", momentum="saved", grams="saved")]
    for replica in pools:
        for momentum, grams in recipe["combinations"]:
            if momentum == grams == "saved":
                continue
            conditions.append(
                dict(name=f"{replica}-M{momentum}-G{grams}", replica=replica, momentum=momentum, grams=grams)
            )

    def optimizer_state(condition):
        if condition["replica"] == "saved":
            return state.opt_state
        candidate = jax.tree.map(lambda a, b: jax.device_put(a, b.sharding), pools[condition["replica"]], saved)
        history = combine_histories(saved, candidate, momentum=condition["momentum"], grams=condition["grams"])
        return replace_matrix_history(state.opt_state, history)

    def update(grad, condition, alpha):
        proposed, post = functions["full", alpha](
            grad, optimizer_state(condition), eqx.filter(native_model, eqx.is_inexact_array)
        )
        norm = float(optax.global_norm(proposed))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid history-control update")
        direction = tree_scale(jax.tree.structure(model).unflatten(jax.tree.leaves(proposed)), -1 / norm)
        return direction, norm, post

    norms = []
    for batch in panels["update"]:
        _, grad = native_grad(native_model, batch)
        direction, norm, post = update(grad, conditions[0], 0.125)
        norms.append(norm)
        del direction, post, grad
    r = float(np.median(norms))
    grid, radius = [f * r for f in protocol["length_grid"]], 4 * r
    calibration, cal_weights = panels["calibration"], weights["calibration"]
    engine = PathMeasurement(model, calibration, cal_weights, [len(calibration)], value_grad, hvp, loss_fn, progress)
    baseline, gradient = engine.baseline[len(calibration)]
    output = dict(
        state=args.state,
        counter=int(state.step),
        data_manifest_sha256=manifest_hash,
        history_records=records,
        conditions=conditions,
        reference_norm=r,
        radius=radius,
        calibration_baseline=baseline,
        calibration_weights=cal_weights,
        evaluation_weights=weights["evaluation"],
        rows=[],
    )
    for draw, batch in enumerate(panels["update"]):
        _, grad = native_grad(native_model, batch)
        for condition in conditions:
            first_history = None
            for alpha in (0.125, 0.25):
                started = time.perf_counter()
                direction, norm, post = update(grad, condition, alpha)
                native_seconds = time.perf_counter() - started
                advanced = describe_arrays(matrix_state(native_model, post).optimizer)
                if first_history is None:
                    first_history = advanced
                elif first_history != advanced:
                    raise ValueError("History advanced differently across inverse powers")
                del post
                product = mean_product(model, direction, calibration, cal_weights, hvp)
                alignment, curvature = float(tree_dot(gradient, direction)), float(tree_dot(direction, product))
                if draw == 0 and condition["name"] == "saved":
                    if alpha == 0.125:
                        check = engine.derivative_check(direction, 0)
                        output["HVP_validation"] = check
                        save("history-partial.json", output)
                        if not check["passed"]:
                            raise ValueError("History-control HVP validation failed")
                        first_direction, first_product = [
                            tuple(np.array(host_array(x), copy=True) for x in jax.tree.leaves(vector))
                            for vector in (direction, product)
                        ]
                    else:
                        symmetry = abs(
                            float(
                                host_inner_product(first_direction, product)
                                - host_inner_product(first_product, direction)
                            )
                        )
                        symmetry /= max(tree_norm(product), host_vector_norm(first_product), 1e-30)
                        output["HVP_symmetry_relative_error"] = symmetry
                        if symmetry >= 1e-4:
                            save("history-partial.json", output)
                            raise ValueError("History-control HVP symmetry failed")
                        del first_direction, first_product
                step, predicted = quadratic_step(alignment, curvature, radius)
                cache = {0.0: baseline}
                for length in grid:
                    if length not in cache:
                        cache[length] = [float(loss_fn(tree_add(model, direction, -length), b)) for b in calibration]
                refined, boundary = refinement_lengths(grid, [np.average(cache[s], weights=cal_weights) for s in grid])
                for length in refined:
                    cache[length] = [float(loss_fn(tree_add(model, direction, -length), b)) for b in calibration]
                chosen = min(cache, key=lambda s: (np.average(cache[s], weights=cal_weights), s))
                output["rows"].append(
                    dict(
                        draw=draw,
                        condition=condition["name"],
                        alpha=alpha,
                        native_norm=norm,
                        native_seconds=native_seconds,
                        alignment=alignment,
                        curvature=curvature,
                        H_product_norm=tree_norm(product),
                        quadratic_step=step,
                        predicted_decrease=predicted,
                        direct_step=chosen,
                        grid_boundary_winner=boundary,
                        direct_at_radius=chosen == radius,
                        direct_at_zero=chosen == 0,
                        calibration=[
                            dict(length=s, losses=v, mean=float(np.average(v, weights=cal_weights)))
                            for s, v in sorted(cache.items())
                        ],
                    )
                )
                save("history-partial.json", output)
                progress(dict(event="history_calibration", draw=draw, condition=condition["name"], alpha=alpha))
                del direction, product
            del first_history
        del grad
    locked = json.loads(json.dumps(output))
    save("locked-predictions.json", locked)
    output["locked_predictions_sha256"] = hashlib.sha256(json.dumps(locked, sort_keys=True).encode()).hexdigest()
    progress(dict(event="predictions_locked", rows=len(output["rows"]), sha256=output["locked_predictions_sha256"]))
    evaluation = panels["evaluation"]
    baseline_eval = [float(loss_fn(model, b)) for b in evaluation]
    output["evaluation_baseline"] = baseline_eval
    for draw, batch in enumerate(panels["update"]):
        _, grad = native_grad(native_model, batch)
        for condition in conditions:
            for alpha in (0.125, 0.25):
                direction, norm, post = update(grad, condition, alpha)
                row = next(
                    r
                    for r in output["rows"]
                    if (r["draw"], r["condition"], r["alpha"]) == (draw, condition["name"], alpha)
                )
                if not np.isclose(norm, row["native_norm"], rtol=1e-6):
                    raise ValueError("Regenerated history direction changed")
                cache = {0.0: baseline_eval}
                row["evaluation"] = {}
                lengths = [
                    ("native", norm),
                    ("equal", r),
                    ("direct", row["direct_step"]),
                    ("quadratic", row["quadratic_step"]),
                ]
                for name, length in lengths:
                    candidate = tree_add(model, direction, -length)
                    if length not in cache:
                        cache[length] = [float(loss_fn(candidate, b)) for b in evaluation]
                    row["evaluation"][name] = dict(
                        length=length,
                        losses=cache[length],
                        loss_change=float(
                            np.average(np.asarray(cache[length]) - baseline_eval, weights=weights["evaluation"])
                        ),
                        predicted_change=-length * row["alignment"] + 0.5 * length**2 * row["curvature"],
                        actual_norm=tree_norm(tree_add(candidate, model, -1)),
                    )
                    del candidate
                save("history-results.json", output)
                progress(dict(event="history_heldout", draw=draw, condition=condition["name"], alpha=alpha))
                del direction, post
        del grad
    return output
