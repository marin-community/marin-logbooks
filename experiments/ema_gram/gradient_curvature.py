# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paired full-model FP32 curvature along complete production optimizer directions."""

import argparse
import hashlib
import json
import os
import time
import uuid
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from google.cloud import storage
from iris.jax.init import initialize_jax
from levanter.checkpoint import load_checkpoint
from levanter.data.loader import DataLoader
from levanter.layers.attention import AttentionBackend
from levanter.models.loss import next_token_loss
from levanter.optim.muon import MuonConfig
from levanter.trainer_state import TrainerState
from levanter.utils.tree_utils import inference_mode
from rigging.secrets import require_secret

from experiments.ema_gram.gradient_curvature_math import (
    predicted_loss,
    tree_add,
    tree_dot,
    tree_norm,
    tree_scale,
    weighted_gradient,
)
from experiments.ema_gram.hessian import FD_STEPS, host_array
from experiments.ema_gram.probe import resolve_config


def measurement_functions(axis_mapping):
    def reference_loss(model, batch):
        with jax.default_matmul_precision("highest"):
            logits = model(batch.tokens, batch.attn_mask, key=None).astype(jnp.float32)
            return next_token_loss(
                model.Pos, model.Vocab, logits, batch.tokens, loss_weight=batch.loss_weight, logsumexp_weight=1e-7
            ).scalar()

    def product(model, direction, batch):
        with jax.default_matmul_precision("highest"):
            return jax.jvp(lambda m: eqx.filter_grad(reference_loss)(m, batch), (model,), (direction,))[1]

    return (
        hax.named_jit(eqx.filter_value_and_grad(reference_loss), axis_resources=axis_mapping),
        hax.named_jit(product, axis_resources=axis_mapping),
        hax.named_jit(reference_loss, axis_resources=axis_mapping),
    )


def reference_data(cfg, seed, update_seeds, save):
    """Use the same deterministic 128-sequence superset and four update draws in all states."""
    seen, manifest, rejected = set(), [], []

    def batches(data_seed, count, role):
        data = cfg.data.train_set(cfg.model.max_Pos, cfg.trainer.batch_schedule, key=jax.random.PRNGKey(data_seed))
        loader = DataLoader(
            data,
            batch_size=hax.Axis("batch", cfg.trainer.train_batch_size),
            axis_resources=cfg.trainer.compute_axis_mapping,
        )
        selected = []
        for source_index, batch in enumerate(loader):
            tokens = host_array(batch.tokens.array)
            hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in tokens]
            if len(set(hashes)) != len(hashes) or any(h in seen for h in hashes):
                rejected.append(dict(role=role, seed=data_seed, source_index=source_index))
                continue
            seen.update(hashes)
            weight = float(host_array(batch.loss_weight.array)[..., :-1].sum())
            manifest.append(
                dict(
                    role=role,
                    seed=data_seed,
                    source_index=source_index,
                    group=len(selected),
                    valid_weight=weight,
                    sequence_hashes=hashes,
                )
            )
            selected.append(batch)
            if len(selected) == count:
                return selected
        raise ValueError("Insufficient unique data")

    reference = batches(seed, 16, "reference")
    updates = [batches(s, 1, "update")[0] for s in update_seeds]
    save(
        "data-manifest.json",
        dict(
            records=manifest,
            rejected=rejected,
            reference_seed=seed,
            update_seeds=update_seeds,
            scope="No duplicate complete sequences within this study; original training exposure not excluded",
        ),
    )
    return reference, updates, [r["valid_weight"] for r in manifest[:16]]


def production_functions(cfg, alphas=(0.125, 0.25), modes=("full", "matrix_only")):
    def loss(m, b):
        m = cfg.trainer.mp.cast_to_compute(m)
        return m.compute_next_token_loss(b, key=None, logsumexp_weight=cfg.z_loss_weight).scalar()

    value_grad = hax.named_jit(eqx.filter_value_and_grad(loss), axis_resources=cfg.trainer.compute_axis_mapping)
    update_functions = {}
    for mode in modes:
        for alpha in alphas:
            optimizer_cfg = replace(
                cfg.optimizer, alpha=alpha, weight_decay=0.0 if mode == "matrix_only" else cfg.optimizer.weight_decay
            )
            optimizer = optimizer_cfg.build(cfg.trainer.num_train_steps)
            update_functions[mode, alpha] = hax.named_jit(
                optimizer.update, axis_resources=cfg.trainer.compute_axis_mapping
            )
    return value_grad, update_functions


def production_updates(state, model, gradient, update_fn, matrix_only):
    """Apply one moment update to the same saved optimizer state in every branch."""
    updates, next_state = update_fn(gradient, state.opt_state, eqx.filter(model, eqx.is_inexact_array))
    jax.block_until_ready(updates)
    del next_state
    if matrix_only:
        mask = MuonConfig().create_mask(model, use_kimi_scaling=True)
        updates = jax.tree.map(lambda label, x: x if label == "muon" else jax.tree.map(jnp.zeros_like, x), mask, updates)
    norm = float(optax.global_norm(updates))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid production update norm")
    return updates, norm


class PathMeasurement:
    """Cache scalar path evidence while keeping only baseline gradients on device."""

    def __init__(self, model, reference, weights, panel_sizes, value_grad, hvp, loss_fn, progress):
        self.model, self.reference, self.weights = model, reference, weights
        self.panel_sizes, self.value_grad, self.hvp, self.loss_fn = panel_sizes, value_grad, hvp, loss_fn
        self.progress = progress
        self.baseline = {}
        for count in panel_sizes:
            losses, gradient = weighted_gradient(model, reference[:count], weights[:count], value_grad)
            self.baseline[count] = (losses, gradient)
            progress(
                dict(
                    event="reference_baseline",
                    groups=count,
                    loss=float(np.average(losses, weights=weights[:count])),
                    gradient_norm=tree_norm(gradient),
                )
            )

    def point(self, direction, length):
        candidate = tree_add(self.model, direction, -length)
        actual = tree_add(candidate, self.model, -1.0)
        actual_norm = tree_norm(actual)
        rounding = tree_norm(tree_add(actual, direction, length)) / max(length, 1e-30)
        result = {}
        for count in self.panel_sizes:
            batches, weights = self.reference[:count], self.weights[:count]
            losses, gradient = weighted_gradient(candidate, batches, weights, self.value_grad)
            product_mean, kappas = None, []
            for batch, weight in zip(batches, np.asarray(weights) / np.sum(weights), strict=True):
                product = self.hvp(candidate, direction, batch)
                kappas.append(float(tree_dot(direction, product)))
                product_mean = (
                    tree_scale(product, weight) if product_mean is None else tree_add(product_mean, product, weight)
                )
            baseline_losses, baseline_gradient = self.baseline[count]
            result[str(count)] = dict(
                groups=count,
                loss=float(np.average(losses, weights=weights)),
                losses=losses,
                loss_change=float(np.average(np.asarray(losses) - baseline_losses, weights=weights)),
                G=tree_norm(gradient),
                alignment=float(tree_dot(gradient, direction)),
                kappa=float(tree_dot(direction, product_mean)),
                Ldir=tree_norm(product_mean),
                group_kappa=kappas,
                gradient_change_rate=tree_norm(tree_add(gradient, baseline_gradient, -1.0)) / length if length else None,
                proposed_norm=length,
                actual_norm=actual_norm,
                rounding_relative_error=rounding,
            )
        return result

    def derivative_check(self, direction, length):
        candidate = tree_add(self.model, direction, -length)
        count = 8
        batches, weights = self.reference[:count], self.weights[:count]
        product_mean = None
        for batch, weight in zip(batches, np.asarray(weights) / np.sum(weights), strict=True):
            product = self.hvp(candidate, direction, batch)
            product_mean = (
                tree_scale(product, weight) if product_mean is None else tree_add(product_mean, product, weight)
            )
        norm = tree_norm(product_mean)
        checks, last_ok = [], False
        for delta in FD_STEPS:
            _, plus = weighted_gradient(tree_add(candidate, direction, delta), batches, weights, self.value_grad)
            _, minus = weighted_gradient(tree_add(candidate, direction, -delta), batches, weights, self.value_grad)
            estimate = tree_scale(tree_add(plus, minus, -1.0), 1 / (2 * delta))
            error = tree_norm(tree_add(estimate, product_mean, -1.0)) / max(norm, 1e-30)
            recorded_error = error if np.isfinite(error) else None
            checks.append(dict(perturbation=delta, relative_error=recorded_error))
            self.progress(
                dict(event="derivative_check", path_length=length, perturbation=delta, relative_error=recorded_error)
            )
            if error < 0.02 and last_ok:
                return dict(passed=True, path_length=length, hvp_norm=norm, tolerance=0.02, checks=checks)
            last_ok = error < 0.02
        return dict(passed=False, path_length=length, hvp_norm=norm, tolerance=0.02, checks=checks)


def path_recorder(engine, directions, reference_length, metadata, output, draw, mode, update_seed, save, progress):
    points = {0.125: {}, 0.25: {}}

    def measure(alpha, fraction):
        key = round(float(fraction), 12)
        if key not in points[alpha]:
            measured = engine.point(directions[alpha], reference_length * key)
            points[alpha][key] = measured
            for values in measured.values():
                output["rows"].append(
                    dict(
                        draw=draw,
                        update_seed=update_seed,
                        mode=mode,
                        alpha=alpha,
                        reference_length=reference_length,
                        fraction=key,
                        **metadata[alpha],
                        **values,
                    )
                )
            save("path-results.json", output)
            progress(
                dict(
                    event="path_point",
                    draw=draw,
                    mode=mode,
                    alpha=alpha,
                    fraction=key,
                    loss_change=measured["8"]["loss_change"],
                    G=measured["8"]["G"],
                    Ldir=measured["8"]["Ldir"],
                )
            )
        return points[alpha][key]

    return measure


def run_paths(state, model, native_model, cfg, reference, updates, weights, args, save, progress):
    value_grad, hvp, loss_fn = measurement_functions(cfg.trainer.compute_axis_mapping)
    panel_sizes = [8, 16] if args.double_reference else [8]
    engine = PathMeasurement(model, reference, weights, panel_sizes, value_grad, hvp, loss_fn, progress)
    output = dict(
        state=args.state,
        counter=int(state.step),
        trajectory=args.trajectory,
        checkpoint=args.checkpoint,
        source_commit=os.environ.get("GIT_COMMIT"),
        reference_weights=weights,
        baseline={str(k): v[0] for k, v in engine.baseline.items()},
        rows=[],
        derivative_checks=[],
        comparisons=[],
    )
    modes = ["full", "matrix_only"] if args.middle else ["full"]
    definition = jax.tree.structure(model)
    production_grad, update_functions = production_functions(cfg)
    for draw, batch in enumerate(updates):
        gradient_loss, gradient = production_grad(native_model, batch)
        gradient_norm = float(optax.global_norm(gradient))
        gradient_info = dict(
            update_loss=float(gradient_loss),
            update_unclipped_gradient_norm=gradient_norm,
            update_clipping_factor=min(1.0, cfg.optimizer.max_grad_norm / max(gradient_norm, 1e-30)),
        )
        for mode in modes:
            directions, norms, metadata = {}, {}, {}
            for alpha in (0.125, 0.25):
                update, norm = production_updates(
                    state, native_model, gradient, update_functions[mode, alpha], mode == "matrix_only"
                )
                info = dict(native_norm=norm, **gradient_info)
                # Reuse the arrays with the vanilla-attention static measurement model.
                update = definition.unflatten(jax.tree.leaves(update))
                directions[alpha] = tree_scale(update, -1 / info["native_norm"])
                norms[alpha], metadata[alpha] = info["native_norm"], info
                progress(dict(event="production_direction", draw=draw, mode=mode, alpha=alpha, **info))

            reference_length = norms[0.125]

            measure = path_recorder(
                engine,
                directions,
                reference_length,
                metadata,
                output,
                draw,
                mode,
                args.update_seeds[draw],
                save,
                progress,
            )

            if draw == 0:
                for alpha in (0.125, 0.25):
                    check = engine.derivative_check(directions[alpha], 0.0)
                    output["derivative_checks"].append(dict(mode=mode, alpha=alpha, draw=draw, **check))
                    save("path-results.json", output)
                    if not check["passed"]:
                        raise ValueError("Representative initial HVP failed numerical validation")
            for alpha in (0.125, 0.25):
                for fraction in (0, 0.05, 0.1, 0.15, 0.3, 0.5, 1.0):
                    measure(alpha, fraction)
            if draw == 0 and mode == "full":
                for alpha in (0.125, 0.25):
                    check = engine.derivative_check(directions[alpha], reference_length)
                    output["derivative_checks"].append(dict(mode=mode, alpha=alpha, draw=draw, **check))
                    save("path-results.json", output)
                    if not check["passed"]:
                        raise ValueError("Representative endpoint HVP failed numerical validation")
            for fraction in (0.1, 0.3, 1.0):
                for intervals in (2, 4, 8, 16):
                    predictions = {}
                    for alpha in (0.125, 0.25):
                        values = [measure(alpha, fraction * u)["8"]["kappa"] for u in np.linspace(0, 1, intervals + 1)]
                        predictions[alpha] = predicted_loss(
                            reference_length * fraction, measure(alpha, 0)["8"]["alignment"], values
                        )
                    observed = measure(0.125, fraction)["8"]["loss_change"] - measure(0.25, fraction)["8"]["loss_change"]
                    predicted = predictions[0.125] - predictions[0.25]
                    error = abs(predicted - observed)
                    resolved = error <= max(2e-6, 0.2 * abs(observed))
                    comparison = dict(
                        draw=draw,
                        mode=mode,
                        fraction=fraction,
                        intervals=intervals,
                        observed_eighth_minus_quarter=observed,
                        predicted_eighth_minus_quarter=predicted,
                        prediction_error=error,
                        quadrature_resolved=resolved,
                        predicted_changes={str(k): v for k, v in predictions.items()},
                    )
                    if resolved or intervals == 16:
                        output["comparisons"].append(comparison)
                        save("path-results.json", output)
                        break
            del directions, measure
    return output


def run_adaptive(state, model, native_model, cfg, reference, updates, weights, args, save, progress):
    """Direct loss evaluations at predeclared calibration or held-out physical lengths."""
    plan = json.loads(args.adaptive_plan.read_text())
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    expected = plan["states"][args.state]
    if expected["counter"] != int(state.step) or expected["trajectory"] != args.trajectory:
        raise ValueError("Adaptive plan checkpoint mismatch")
    if expected["data_manifest_sha256"] != args.data_manifest_sha256:
        raise ValueError("Reference/update data changed between path and adaptive phases")
    tasks = plan["tasks"][args.state]
    if any((t["heldout"] == args.state) != (plan["stage"] == "heldout") for t in tasks):
        raise ValueError("Calibration and held-out tasks must remain separate")
    value_grad, _, loss_fn = measurement_functions(cfg.trainer.compute_axis_mapping)
    baseline, reference_gradient = weighted_gradient(model, reference[:8], weights[:8], value_grad)
    g = tree_norm(reference_gradient)
    if not np.isclose(g, expected["G"], rtol=1e-4, atol=1e-6):
        raise ValueError("Reference gradient changed between path and adaptive phases")
    output = dict(
        state=args.state,
        counter=int(state.step),
        trajectory=args.trajectory,
        checkpoint=args.checkpoint,
        source_commit=os.environ.get("GIT_COMMIT"),
        plan_hash=plan_hash,
        stage=plan["stage"],
        reference_weights=weights[:8],
        baseline=baseline,
        G=g,
        rows=[],
    )
    save("adaptive-plan.json", plan)
    production_grad, update_functions = production_functions(cfg)
    definition = jax.tree.structure(model)
    for draw, batch in enumerate(updates):
        update_loss, gradient = production_grad(native_model, batch)
        gradient_norm = float(optax.global_norm(gradient))
        for alpha in (0.125, 0.25):
            update, native_norm = production_updates(
                state, native_model, gradient, update_functions["full", alpha], False
            )
            direction = tree_scale(definition.unflatten(jax.tree.leaves(update)), -1 / native_norm)
            alignment = float(tree_dot(reference_gradient, direction))
            cache = {}
            for task in tasks:
                length = task["length"]
                if length not in cache:
                    candidate = tree_add(model, direction, -length)
                    displacement = tree_add(candidate, model, -1.0)
                    losses = [float(loss_fn(candidate, b)) for b in reference[:8]]
                    cache[length] = dict(
                        losses=losses,
                        loss=float(np.average(losses, weights=weights[:8])),
                        loss_change=float(np.average(np.asarray(losses) - baseline, weights=weights[:8])),
                        actual_norm=tree_norm(displacement),
                        rounding_relative_error=tree_norm(tree_add(displacement, direction, length)) / length,
                    )
                output["rows"].append(
                    dict(
                        **task,
                        **cache[length],
                        draw=draw,
                        update_seed=args.update_seeds[draw],
                        alpha=alpha,
                        G=g,
                        alignment=alignment,
                        nonpositive_alignment=alignment <= 0,
                        native_norm=native_norm,
                        update_loss=float(update_loss),
                        update_unclipped_gradient_norm=gradient_norm,
                        update_clipping_factor=min(1.0, cfg.optimizer.max_grad_norm / max(gradient_norm, 1e-30)),
                    )
                )
                save("adaptive-results.json", output)
                progress(
                    dict(
                        event="adaptive_evaluation",
                        stage=plan["stage"],
                        draw=draw,
                        alpha=alpha,
                        rule=task["rule"],
                        heldout=task["heldout"],
                        length=length,
                        loss_change=cache[length]["loss_change"],
                    )
                )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--config-index", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-counter", type=int, required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--trajectory", type=float, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--middle", action="store_true")
    parser.add_argument("--double-reference", action="store_true")
    parser.add_argument("--adaptive-plan", type=Path)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--update-seeds", type=int, nargs=4, default=[20262112, 20262113, 20262114, 20262115])
    parser.add_argument("--logging-check", action="store_true")
    args = parser.parse_args()
    encoded = json.loads(args.configs.read_text())[args.config_index]["config"]["train_config"]
    cfg = resolve_config(encoded)
    if cfg.optimizer.alpha != args.trajectory or cfg.z_loss_weight != 1e-7 or cfg.trainer.train_batch_size != 8:
        raise ValueError("Configuration does not match declared protocol")
    identity = "ema-gram-gradient-curvature-" + uuid.uuid4().hex[:12]
    local = Path(os.environ.get("IRIS_OUTPUT_DIR", "scratch/gradient-curvature")) / identity
    local.mkdir(parents=True)
    run = wandb.init(
        entity="marin-community",
        project="marin",
        id=identity,
        name=identity,
        group="ema-gram-20-gradient-curvature",
        mode="online",
        resume="never",
        settings={"api_key": require_secret("WANDB_API_KEY")},
        config=dict(
            resolved_training_config=encoded,
            source_commit=os.environ.get("GIT_COMMIT"),
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        ),
    )
    if run.settings.mode != "online":
        raise RuntimeError("Online W&B required")
    run.log({"logging_check/success": 1}, step=0)
    print(json.dumps(dict(event="tracking_started", url=run.url)), flush=True)
    if args.logging_check:
        run.finish()
        return
    initialize_jax()
    for key, value in cfg.trainer.jax_config.items():
        jax.config.update(key, value)
    bucket_name, prefix = args.output_prefix.removeprefix("gs://").rstrip("/").split("/", 1)
    bucket = storage.Client().bucket(bucket_name)
    prefix += "/" + identity
    step = 0
    started = time.perf_counter()

    def save(name, data):
        path = local / name
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        if jax.process_index() == 0:
            bucket.blob(prefix + "/" + name).upload_from_filename(str(path))

    def progress(data):
        nonlocal step
        step += 1
        data = dict(step=step, elapsed_seconds=time.perf_counter() - started, **data)
        with (local / "measurements.jsonl").open("a") as f:
            f.write(json.dumps(data, allow_nan=False) + "\n")
        run.log(data, step=step)
        print(json.dumps(data), flush=True)
        if jax.process_index() == 0:
            bucket.blob(prefix + "/measurements.jsonl").upload_from_filename(str(local / "measurements.jsonl"))

    try:
        with cfg.trainer.use_device_mesh():
            vocab = hax.partitioning.round_axis_for_partitioning(
                hax.Axis("vocab", len(cfg.data.the_tokenizer)), cfg.trainer.compute_axis_mapping
            )
            optimizer = cfg.optimizer.build(cfg.trainer.num_train_steps)
            template = eqx.filter_eval_shape(
                lambda: TrainerState.init(
                    optimizer,
                    cfg.model.build(vocab, key=jax.random.PRNGKey(0)),
                    key=jax.random.PRNGKey(0),
                    mp=cfg.trainer.mp,
                )
            )
            state = load_checkpoint(
                template, args.checkpoint, axis_mapping=cfg.trainer.parameter_axis_mapping, allow_partial=False
            )
            if int(state.step) != args.expected_counter:
                raise ValueError("Complete restore counter mismatch")
            native_model = inference_mode(state.model, True)
            measurement_config = replace(cfg.model, attn_backend=AttentionBackend.VANILLA)
            shape = inference_mode(
                eqx.filter_eval_shape(lambda: measurement_config.build(vocab, key=jax.random.PRNGKey(0))), True
            )
            model = jax.tree.structure(shape).unflatten(jax.tree.leaves(native_model))
            if any(x.dtype != jnp.float32 for x in jax.tree.leaves(model) if eqx.is_array(x)):
                raise ValueError("Reference parameters must be FP32")
            save(
                "configuration.json",
                dict(
                    training=encoded,
                    checkpoint=args.checkpoint,
                    counter=int(state.step),
                    source_commit=os.environ.get("GIT_COMMIT"),
                    reference_precision="FP32, vanilla attention, unfused CE, highest matmul",
                    production_precision=str(cfg.trainer.mp),
                    reference_seed=args.seed,
                    update_seeds=args.update_seeds,
                    common_learning_rate="Original checkpoint schedule shared across candidate powers",
                    checkpoint_resume="Full saved state restored; no measurement progress resumed",
                    host_count=jax.process_count(),
                    devices=jax.device_count(),
                ),
            )
            progress(dict(event="restored", counter=int(state.step)))
            reference, updates, weights = reference_data(cfg, args.seed, args.update_seeds, save)
            args.data_manifest_sha256 = hashlib.sha256((local / "data-manifest.json").read_bytes()).hexdigest()
            runner = run_adaptive if args.adaptive_plan else run_paths
            result = runner(state, model, native_model, cfg, reference, updates, weights, args, save, progress)
            save(
                "completed.json",
                dict(
                    success=True,
                    state=args.state,
                    counter=int(state.step),
                    rows=len(result["rows"]),
                    elapsed_seconds=time.perf_counter() - started,
                    output="gs://" + bucket_name + "/" + prefix,
                    wandb=run.url,
                    host_count=jax.process_count(),
                    device_count=jax.device_count(),
                ),
            )
    except BaseException as error:
        save("failure.json", dict(error=repr(error), wandb=run.url, elapsed_seconds=time.perf_counter() - started))
        run.finish(exit_code=1)
        raise
    else:
        run.finish()


if __name__ == "__main__":
    main()
