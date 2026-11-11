# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Disjoint fixed-weight Gram histories and paired full-optimizer loss probes."""

import hashlib
from dataclasses import replace
from itertools import islice

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from levanter.data.loader import DataLoader
from levanter.optim.ema_gram import GramState, update_gram_state
from levanter.optim.util import flatten_linear_layers


def rank_one_factors(directions):
    """Recover unit left/right factors without constructing a Kronecker matrix."""
    left, right = [], []
    for direction in directions.astype(np.float64):
        column = int(np.argmax(np.linalg.norm(direction, axis=0)))
        u = direction[:, column] / np.linalg.norm(direction[:, column])
        v = u @ direction
        if not np.allclose(np.outer(u, v), direction, rtol=1e-5, atol=1e-7):
            raise ValueError("Expected unit rank-one basis directions")
        left.append(u)
        right.append(v)
    return np.stack(left).T, np.stack(right).T


def project_grams(left, right, u, v, ridge=0.01):
    """Keep all projected entries, including rotations away from historical modes."""
    raw = (u.T @ left @ u) * (v.T @ right @ v)
    lm, rm = np.trace(left) / len(left), np.trace(right) / len(right)
    if min(lm, rm) <= 0:
        raise ValueError("Fresh Gram normalization requires positive traces")
    a = left / lm + ridge * np.eye(len(left))
    b = right / rm + ridge * np.eye(len(right))
    return raw, (u.T @ a @ u) * (v.T @ b @ v)


def effective_ema_samples(beta, count):
    weights = (1 - beta) * beta ** np.arange(count)
    return float(weights.sum() ** 2 / np.sum(weights**2))


def run_controls(
    state,
    model,
    cfg,
    blocks,
    save,
    progress,
    host_array,
    *,
    histories=2,
    history_batches=512,
    curvature_seed=20261110,
    curvature_groups=8,
):
    """Keep every candidate rooted in the same complete checkpoint, without training."""
    if histories < 2 or history_batches < 2:
        raise ValueError("Use at least two histories and two batches")
    seen = set()
    manifest = []
    rejections = []

    def record(batch, role, index, seed, source_index=None):
        tokens = host_array(batch.tokens.array)
        hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in tokens]
        if len(set(hashes)) != len(hashes) or any(h in seen for h in hashes):
            rejections.append(dict(role=role, source_index=source_index, seed=seed, reason="Duplicate sequence"))
            return False
        seen.update(hashes)
        weights = host_array(batch.loss_weight.array)
        manifest.append(
            dict(
                role=role,
                index=index,
                source_index=source_index,
                seed=seed,
                token_sha256=hashes,
                tokens=int(tokens.size),
                valid_weight=float(weights[..., :-1].sum()),
            )
        )
        return True

    def loader(seed):
        data = cfg.data.train_set(cfg.model.max_Pos, cfg.trainer.batch_schedule, key=jax.random.PRNGKey(seed))
        return DataLoader(
            data,
            batch_size=hax.Axis("batch", cfg.trainer.train_batch_size),
            axis_resources=cfg.trainer.compute_axis_mapping,
        )

    def unique_batches(seed, count, role):
        accepted = 0
        for source_index, batch in enumerate(loader(seed)):
            if record(batch, role, accepted, seed, source_index):
                yield batch
                accepted += 1
                if accepted == count:
                    return
        raise ValueError(f"Insufficient disjoint batches for {role}")

    def training_loss(m, batch):
        m = cfg.trainer.mp.cast_to_compute(m)
        return m.compute_next_token_loss(batch, key=None, logsumexp_weight=cfg.z_loss_weight).scalar()

    value_grad = hax.named_jit(eqx.filter_value_and_grad(training_loss), axis_resources=cfg.trainer.compute_axis_mapping)
    loss_fn = hax.named_jit(training_loss, axis_resources=cfg.trainer.compute_axis_mapping)
    block_paths = [b["path"] for b in blocks]

    @hax.named_jit(axis_resources=cfg.trainer.compute_axis_mapping)
    def fresh_step(m, batch, grams):
        loss, gradient = eqx.filter_value_and_grad(training_loss)(m, batch)
        norm = optax.global_norm(gradient)
        # Same global clipping convention as the actual optimizer.
        clipped, _ = optax.clip_by_global_norm(cfg.optimizer.max_grad_norm).update(gradient, ())
        flat = {
            jax.tree_util.keystr(p): x
            for p, x in jax.tree_util.tree_flatten_with_path(flatten_linear_layers(clipped))[0]
        }
        updated = []
        for path, gram in zip(block_paths, grams, strict=True):
            g = flat[path].reshape((-1, *gram.momentum.shape))[0]
            updated.append(update_gram_state(g, gram, cfg.optimizer.momentum, cfg.optimizer.gram_beta))
        return loss, norm, tuple(updated)

    # Explicitly exclude the earlier curvature groups by checking their fingerprints.
    for i, batch in enumerate(islice(loader(curvature_seed), curvature_groups)):
        if not record(batch, "curvature_exclusion", i, curvature_seed, i):
            raise ValueError("Duplicated original curvature data")

    results = []
    for history in range(histories):
        seed = 20261210 + history
        grams = tuple(
            GramState(
                jnp.zeros(b["shape"], jnp.float32),
                jnp.zeros((b["shape"][0], b["shape"][0]), jnp.float32),
                jnp.zeros((b["shape"][1], b["shape"][1]), jnp.float32),
            )
            for b in blocks
        )
        losses, norms = [], []
        for i, batch in enumerate(unique_batches(seed, history_batches, f"gram_history_{history}")):
            loss, norm, grams = fresh_step(model, batch, grams)
            losses.append(float(loss))
            norms.append(float(norm))
            if not np.isfinite([losses[-1], norms[-1]]).all():
                raise ValueError("Nonfinite frozen gradient measurement")
            if (i + 1) % 32 == 0:
                progress(
                    dict(event="fresh_grams", history=history, batches=i + 1, loss=losses[-1], gradient_norm=norms[-1])
                )
                save("controls-data-manifest.json", manifest)
                save("controls-rejected-batches.json", rejections)
        if len(losses) != history_batches:
            raise ValueError("Fresh history ended early")
        correction = 1 - cfg.optimizer.gram_beta**history_batches
        for block, gram in zip(blocks, grams, strict=True):
            left, right = (
                host_array(gram.left).astype(np.float64) / correction,
                host_array(gram.right).astype(np.float64) / correction,
            )
            u, v = rank_one_factors(block["directions"])
            raw, ridged = project_grams(left, right, u, v, cfg.optimizer.ridge)
            results.append(
                dict(
                    history=history,
                    seed=seed,
                    block=block["label"],
                    gram_raw=raw.tolist(),
                    gram_ridged=ridged.tolist(),
                    normalization_means=[float(np.trace(left) / len(left)), float(np.trace(right) / len(right))],
                )
            )
        save(
            "fresh-grams.json",
            dict(
                results=results,
                batches_per_history=history_batches,
                beta=cfg.optimizer.gram_beta,
                bias_correction=correction,
                effective_samples=effective_ema_samples(cfg.optimizer.gram_beta, history_batches),
                gradient_precision=str(cfg.trainer.mp),
                scope="Disjoint randomized training-cache windows at frozen weights; finite EMA is bias corrected",
            ),
        )
        save(
            f"fresh-history-{history}.json",
            dict(
                losses=losses,
                gradient_norms=norms,
                clipped_fraction=float(np.mean(np.array(norms) > cfg.optimizer.max_grad_norm)),
            ),
        )
        del grams

    probe_seed = 20261310
    probe_batches = list(unique_batches(probe_seed, 17, "update_and_evaluation"))
    if len(probe_batches) != 17:
        raise ValueError("Need one update batch and sixteen evaluation batches")
    for row in manifest[-17:]:
        row["role"] = "update_gradient" if row["index"] == 0 else "paired_evaluation"
    save("controls-data-manifest.json", manifest)
    save("controls-rejected-batches.json", rejections)
    update_batch, evaluation = probe_batches[0], probe_batches[1:]
    baseline = [float(loss_fn(model, b)) for b in evaluation]
    gradient_loss, gradient = value_grad(model, update_batch)
    progress(dict(event="update_gradient", loss=float(gradient_loss), gradient_norm=float(optax.global_norm(gradient))))
    rows = []
    reference_norm = None
    for alpha in (0.0, 0.125, 0.25, 0.375, 0.5):
        # Same saved counter, moments, fresh gradient, LR schedule and decay in every branch.
        optimizer = replace(cfg.optimizer, alpha=alpha).build(cfg.trainer.num_train_steps)
        update_fn = hax.named_jit(optimizer.update, axis_resources=cfg.trainer.compute_axis_mapping)
        updates, next_state = update_fn(gradient, state.opt_state, eqx.filter(model, eqx.is_inexact_array))
        jax.block_until_ready(updates)
        del next_state
        norm = float(optax.global_norm(updates))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid full optimizer update")
        if reference_norm is None:
            reference_norm = norm
        for matching, multiplier in (("native", 1.0), ("global_norm_matched_to_alpha0", reference_norm / norm)):
            for fraction in (0.1, 0.3, 1.0):
                scale = multiplier * fraction
                candidate = eqx.apply_updates(model, jax.tree.map(lambda x, scale=scale: scale * x, updates))
                losses = [float(loss_fn(candidate, b)) for b in evaluation]
                delta = np.asarray(losses) - baseline
                if not np.isfinite(delta).all():
                    raise ValueError("Nonfinite paired evaluation loss")
                row = dict(
                    alpha=alpha,
                    matching=matching,
                    step_fraction=fraction,
                    update_norm=norm,
                    applied_update_norm=norm * scale,
                    baseline_losses=baseline,
                    losses=losses,
                    delta_mean=float(delta.mean()),
                    delta_sem=float(delta.std(ddof=1) / np.sqrt(len(delta))),
                )
                rows.append(row)
                save("full-update-losses.json", dict(rows=rows, counter=int(state.step), eval_batches=len(evaluation)))
                progress(
                    dict(
                        event="full_update_loss",
                        alpha=alpha,
                        matching=matching,
                        fraction=fraction,
                        delta_mean=row["delta_mean"],
                    )
                )
                del candidate
        del updates
    save(
        "controls-completed.json",
        dict(
            success=True,
            counter=int(state.step),
            histories=histories,
            history_batches=history_batches,
            update_candidates=len(rows),
            warning="Local counterfactual steps; no training trajectory or causal explanation of full-run ranking",
        ),
    )
