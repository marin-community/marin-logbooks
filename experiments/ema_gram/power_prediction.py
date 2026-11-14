# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lock calibrated power predictions before opening the disjoint evaluation panel."""

import hashlib
import json
import time
from functools import partial

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from levanter.optim.ema_gram import gram_direction, grouped_gram_powers
from levanter.optim.util import unflatten_linear_layers

from experiments.ema_gram.geometry_math import quadratic_step, refinement_lengths
from experiments.ema_gram.gradient_curvature import PathMeasurement, measurement_functions, production_functions
from experiments.ema_gram.gradient_curvature_math import tree_add, tree_dot, tree_norm, tree_scale, weighted_gradient
from experiments.ema_gram.probe import matrix_state

POWERS = (0.125, 0.1875, 0.25)
ALPHA_FD_STEPS = (0.01, 0.003, 0.001, 0.0005, 0.0003, 0.0001, 0.00003)


def fp32_direction_function(cfg):
    """Differentiate alpha with moments frozen; retain the complete native nonmatrix update and decay."""

    def direction(model, post_state, native_middle, alpha):
        masked, definition, moments = matrix_state(model, post_state)
        mesh = jax.sharding.get_abstract_mesh()
        axes = tuple(name for name, size in mesh.shape.items() if size > 1) or ("data",)
        grams = tuple(g for s in moments.matrices for g in (s.left, s.right))
        with jax.default_matmul_precision("highest"):
            changing = grouped_gram_powers(grams, alpha=alpha, ridge=cfg.optimizer.ridge, mesh=mesh, axis_name=axes)
            matrices = []
            for i, s in enumerate(moments.matrices):
                options = dict(alpha=1.0, ridge=cfg.optimizer.ridge, polar_steps=cfg.optimizer.polar_steps)
                a = gram_direction(s, precomputed_powers=changing[2 * i : 2 * i + 2], **options)
                matrices.append(a)
            transformed = unflatten_linear_layers(masked, definition.unflatten(matrices))
            decay = optax.add_decayed_weights(cfg.optimizer.weight_decay, cfg.optimizer.build_weight_decay_mask())
            decayed, _ = decay.update(transformed, decay.init(masked), eqx.filter(masked, eqx.is_inexact_array))
            update = jax.tree.map(
                lambda d, p: p if isinstance(d, optax.MaskedNode) else -post_state.hyperparams["learning_rate"] * d,
                decayed,
                native_middle,
                is_leaf=lambda x: isinstance(x, optax.MaskedNode),
            )
            return tree_scale(update, -1 / jnp.sqrt(tree_dot(update, update)))

    return hax.named_jit(direction, axis_resources=cfg.trainer.compute_axis_mapping)


def mean_product(model, direction, batches, weights, hvp):
    total = None
    for batch, weight in zip(batches, np.asarray(weights) / np.sum(weights), strict=True):
        value = hvp(model, direction, batch)
        total = tree_scale(value, weight) if total is None else tree_add(total, value, weight)
    return total


def run_power_prediction(
    *, state, model, native_model, cfg, panels, weights, args, protocol, save, progress, manifest_hash, save_arrays
):
    del save_arrays
    value_grad, hvp, loss_fn = measurement_functions(cfg.trainer.compute_axis_mapping)
    production_grad, functions = production_functions(cfg, alphas=POWERS, modes=("full",))
    definition = jax.tree.structure(model)
    calibration, cal_weights = panels["calibration"], weights["calibration"]
    reference = PathMeasurement(model, calibration, cal_weights, [8], value_grad, hvp, loss_fn, progress)
    baseline, gradient = reference.baseline[8]
    smooth = fp32_direction_function(cfg)

    def native(grad, alpha):
        update, after = functions["full", alpha](grad, state.opt_state, eqx.filter(native_model, eqx.is_inexact_array))
        jax.block_until_ready(update)
        norm = float(optax.global_norm(update))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid native update")
        return update, after, norm

    # Only eight scalar norms survive this pass, avoiding 24 full models in HBM.
    norms = []
    for draw, batch in enumerate(panels["update"]):
        _, grad = production_grad(native_model, batch)
        update, after, norm = native(grad, 0.125)
        norms.append(norm)
        progress(dict(event="reference_norm", draw=draw, norm=norm))
        del grad, update, after
    radius_reference = float(np.median(norms))
    radius = 4 * radius_reference
    grid = [f * radius_reference for f in protocol["length_grid"]]
    output = dict(
        state=args.state,
        counter=int(state.step),
        data_manifest_sha256=manifest_hash,
        reference_norm=radius_reference,
        radius=radius,
        native_eighth_norms=norms,
        calibration_weights=cal_weights,
        evaluation_weights=weights["evaluation"],
        calibration_baseline=baseline,
        rows=[],
        choices=[],
        derivative_checks=[],
    )

    for draw, batch in enumerate(panels["update"]):
        update_loss, grad = production_grad(native_model, batch)
        first_direction, first_product, numerical_check = None, None, None
        shared_moments = None
        for alpha in POWERS:
            started = time.perf_counter()
            update, after, norm = native(grad, alpha)
            advanced = matrix_state(native_model, after).optimizer
            if shared_moments is None:
                shared_moments = advanced
            elif tree_norm(tree_add(advanced, shared_moments, -1)) != 0:
                raise ValueError("Moments changed across powers")
            direction = tree_scale(definition.unflatten(jax.tree.leaves(update)), -1 / norm)
            product = mean_product(model, direction, calibration, cal_weights, hvp)
            alignment, curvature = float(tree_dot(gradient, direction)), float(tree_dot(direction, product))
            if draw == 0 and alpha == 0.125:
                first_direction, first_product = direction, product
            model_step, predicted = quadratic_step(alignment, curvature, radius)
            cache = {0.0: baseline}

            def losses(length, cache=cache, direction=direction):
                if length not in cache:
                    candidate = tree_add(model, direction, -length)
                    cache[length] = [float(loss_fn(candidate, b)) for b in calibration]
                return cache[length]

            scores = [float(np.average(losses(s), weights=cal_weights)) for s in grid]
            refined, boundary = refinement_lengths(grid, scores)
            for s in refined:
                losses(s)
            direct_step = min(cache, key=lambda s: (np.average(cache[s], weights=cal_weights), s))
            row = dict(
                draw=draw,
                alpha=alpha,
                native_norm=norm,
                alignment=alignment,
                curvature=curvature,
                H_product_norm=tree_norm(product),
                quadratic_step=model_step,
                predicted_decrease=predicted,
                direct_step=direct_step,
                grid_boundary_winner=boundary,
                direct_at_radius=bool(direct_step == radius),
                direct_at_zero=bool(direct_step == 0),
                update_loss=float(update_loss),
                update_gradient_norm=float(optax.global_norm(grad)),
                clipping_factor=min(1.0, cfg.optimizer.max_grad_norm / max(float(optax.global_norm(grad)), 1e-30)),
                calibration=[
                    dict(length=s, losses=v, mean=float(np.average(v, weights=cal_weights)))
                    for s, v in sorted(cache.items())
                ],
                calibration_seconds=time.perf_counter() - started,
            )
            if alpha == 0.1875:
                scalar = jnp.asarray(alpha, jnp.float32)
                # Differentiating only alpha keeps moment construction outside the tangent.
                ds, tangent = jax.jvp(partial(smooth, native_model, after, update), (scalar,), (jnp.ones_like(scalar),))
                ds, tangent = [definition.unflatten(jax.tree.leaves(x)) for x in (ds, tangent)]
                tangent_errors = []
                for epsilon in ALPHA_FD_STEPS:
                    dp = smooth(native_model, after, update, scalar + epsilon)
                    dm = smooth(native_model, after, update, scalar - epsilon)
                    fd = tree_scale(tree_add(dp, dm, -1), 1 / (2 * epsilon))
                    fd = definition.unflatten(jax.tree.leaves(fd))
                    tangent_errors.append(
                        dict(
                            epsilon=epsilon,
                            relative_error=tree_norm(tree_add(fd, tangent, -1)) / max(tree_norm(tangent), 1e-30),
                        )
                    )
                    if len(tangent_errors) > 1 and all(v["relative_error"] < 0.02 for v in tangent_errors[-2:]):
                        break
                row["alpha_derivative"] = dict(
                    alpha=alpha,
                    direction_native_difference=tree_norm(tree_add(ds, direction, -1)),
                    tangent_norm=tree_norm(tangent),
                    normalization_inner_product=float(tree_dot(ds, tangent)),
                    H_direction_dot_alpha_tangent=float(
                        tree_dot(mean_product(model, ds, calibration, cal_weights, hvp), tangent)
                    ),
                    initial_gradient_dot_alpha_tangent=float(tree_dot(gradient, tangent)),
                    finite_difference=tangent_errors,
                    passed=len(tangent_errors) > 1 and all(v["relative_error"] < 0.02 for v in tangent_errors[-2:]),
                )
                if not row["alpha_derivative"]["passed"]:
                    save("power-partial.json", dict(**output, failed_row=row))
                    raise ValueError("FP32 alpha tangent failed two-scale finite differences")
                if draw == 0:
                    # The identity is a fixed-length derivative of the same calibration objective.
                    # We do not claim the derivative of an adaptively changing line-search optimum.
                    smooth_losses = {}
                    for s in grid:
                        candidate = tree_add(model, ds, -s)
                        smooth_losses[s] = [float(loss_fn(candidate, b)) for b in calibration]
                    refined_smooth, _ = refinement_lengths(
                        grid, [np.average(smooth_losses[s], weights=cal_weights) for s in grid]
                    )
                    for s in refined_smooth:
                        smooth_losses[s] = [float(loss_fn(tree_add(model, ds, -s), b)) for b in calibration]
                    chosen = min(smooth_losses, key=lambda s: (np.average(smooth_losses[s], weights=cal_weights), s))
                    endpoint = tree_add(model, ds, -chosen)
                    _, endpoint_gradient = weighted_gradient(endpoint, calibration, cal_weights, value_grad)
                    chain = -chosen * float(tree_dot(endpoint_gradient, tangent))

                    def forward_derivative(m, t, b):
                        return jax.jvp(lambda p: loss_fn(p, b), (m,), (t,))[1]

                    forward = hax.named_jit(forward_derivative, axis_resources=cfg.trainer.compute_axis_mapping)
                    automatic = float(
                        np.average(
                            [float(forward(endpoint, tree_scale(tangent, -chosen), b)) for b in calibration],
                            weights=cal_weights,
                        )
                    )
                    error = abs(automatic - chain) / max(abs(automatic), abs(chain), 1e-12)
                    row["alpha_derivative"]["loss_identity"] = dict(
                        objective="Same 64-sequence calibration CE plus z loss; fixed calibrated physical length",
                        length=chosen,
                        forward_derivative=automatic,
                        gradient_identity=chain,
                        relative_error=error,
                        zero_step=bool(chosen == 0),
                        passed=bool(error < 0.02),
                        smooth_calibration=[dict(length=s, losses=v) for s, v in sorted(smooth_losses.items())],
                    )
                    if error >= 0.02:
                        save("power-partial.json", dict(**output, failed_row=row))
                        raise ValueError("Forward/reverse loss derivative identity failed")
                    del endpoint_gradient, endpoint, forward
                del ds, tangent
            if draw == 0 and alpha == 0.1875:
                numerical_check = reference.derivative_check(direction, 0)
                if not numerical_check["passed"]:
                    save("failed-hvp-check.json", numerical_check)
                    raise ValueError("Reference HVP validation failed")
            if draw == 0 and alpha == 0.25:
                symmetry = abs(float(tree_dot(first_direction, product) - tree_dot(direction, first_product)))
                symmetry /= max(tree_norm(first_product), tree_norm(product), 1e-30)
                output["derivative_checks"].append(dict(**numerical_check, symmetry_relative_error=symmetry))
                if symmetry > 1e-4:
                    save("power-partial.json", output)
                    raise ValueError("Reference HVP symmetry failed")
            output["rows"].append(row)
            save("power-partial.json", output)
            progress(
                dict(
                    event="calibration_direction",
                    draw=draw,
                    alpha=alpha,
                    native_norm=norm,
                    alignment=alignment,
                    curvature=curvature,
                    direct_step=direct_step,
                    quadratic_step=model_step,
                )
            )
            del update, after, direction, product, losses
        rows = [r for r in output["rows"] if r["draw"] == draw]
        output["choices"].append(
            dict(
                draw=draw,
                fixed_eighth=0.125,
                fixed_quarter=0.25,
                alignment=max(rows, key=lambda r: (r["alignment"], -r["alpha"]))["alpha"],
                quadratic=max(rows, key=lambda r: (r["predicted_decrease"], -r["alpha"]))["alpha"],
                calibration_best=min(rows, key=lambda r: (min(v["mean"] for v in r["calibration"]), r["alpha"]))[
                    "alpha"
                ],
            )
        )
        del grad, shared_moments, first_direction, first_product

    # This immutable artifact is uploaded before the first evaluation-panel loss.
    locked = json.loads(json.dumps(output))
    save("locked-predictions.json", locked)
    output["locked_predictions_sha256"] = hashlib.sha256(json.dumps(locked, sort_keys=True).encode()).hexdigest()
    progress(dict(event="predictions_locked", rows=len(output["rows"]), sha256=output["locked_predictions_sha256"]))
    evaluation = panels["evaluation"]
    eval_weights = weights["evaluation"]
    eval_baseline = [float(loss_fn(model, b)) for b in evaluation]
    output["evaluation_baseline"] = eval_baseline
    for draw, batch in enumerate(panels["update"]):
        _, grad = production_grad(native_model, batch)
        for alpha in POWERS:
            update, after, norm = native(grad, alpha)
            direction = tree_scale(definition.unflatten(jax.tree.leaves(update)), -1 / norm)
            row = next(r for r in output["rows"] if (r["draw"], r["alpha"]) == (draw, alpha))
            if not np.isclose(norm, row["native_norm"], rtol=1e-6):
                raise ValueError("Regenerated direction changed")
            cache = {0.0: eval_baseline}
            row["evaluation"] = {}
            evaluation_steps = [
                ("native", norm),
                ("equal", radius_reference),
                ("direct", row["direct_step"]),
                ("quadratic", row["quadratic_step"]),
            ]
            evaluation_steps += [
                (f"equal_{fraction:g}", fraction * radius_reference) for fraction in protocol["length_grid"]
            ]
            for name, length in evaluation_steps:
                start = time.perf_counter()
                candidate = tree_add(model, direction, -length)
                if length not in cache:
                    cache[length] = [float(loss_fn(candidate, b)) for b in evaluation]
                actual_norm = tree_norm(tree_add(candidate, model, -1))
                row["evaluation"][name] = dict(
                    length=length,
                    losses=cache[length],
                    loss_change=float(np.average(np.asarray(cache[length]) - eval_baseline, weights=eval_weights)),
                    predicted_change=-length * row["alignment"] + 0.5 * length**2 * row["curvature"],
                    actual_norm=actual_norm,
                    native_seconds=time.perf_counter() - start,
                )
            save("power-results.json", output)
            progress(
                dict(
                    event="heldout_direction",
                    draw=draw,
                    alpha=alpha,
                    direct_loss_change=row["evaluation"]["direct"]["loss_change"],
                )
            )
            del update, after, direction
        del grad
    return output
