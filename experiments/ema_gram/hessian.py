# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure frozen-checkpoint block Hessians in saved gradient-Gram directions."""

import argparse
import hashlib
import json
import os
import time
import uuid
from dataclasses import replace
from itertools import islice, pairwise
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from google.cloud import storage
from iris.jax.init import initialize_jax
from jax.experimental.multihost_utils import process_allgather
from levanter.checkpoint import load_checkpoint
from levanter.data.loader import DataLoader
from levanter.layers.attention import AttentionBackend
from levanter.models.loss import next_token_loss
from levanter.optim.util import flatten_linear_layers, unflatten_linear_layers
from levanter.trainer_state import TrainerState
from levanter.utils.tree_utils import inference_mode
from rigging.secrets import require_secret

from experiments.ema_gram.probe import matrix_state, resolve_config


def host_array(value):
    return np.asarray(process_allgather(value, tiled=True))


def select_basis(left, right, count=12, rtol=1e-5):
    """Choose distinct products nearest equally spaced resolved log-product targets."""
    eigenvalues = []
    eigenvectors = []
    for gram in (left, right):
        values, vectors = np.linalg.eigh((gram.astype(np.float64) + gram.T) / 2)
        eigenvalues.append(values)
        eigenvectors.append(vectors)
    lv, rv = eigenvalues
    li = np.flatnonzero(lv > rtol * lv[-1])
    ri = np.flatnonzero(rv > rtol * rv[-1])
    if not len(li) or not len(ri):
        raise ValueError("No resolved positive Gram modes")
    products = lv[li, None] * rv[None, ri]
    order = np.argsort(products.ravel(), kind="stable")
    logs = np.log(products.ravel())
    targets = np.linspace(logs.min(), logs.max(), min(count, products.size))
    selected = []
    for target in targets:
        candidates = order[np.argsort(np.abs(logs[order] - target), kind="stable")]
        selected.append(next(int(x) for x in candidates if int(x) not in selected))
    pairs = [(int(li[x // len(ri)]), int(ri[x % len(ri)])) for x in selected]
    directions = np.stack([np.outer(eigenvectors[0][:, i], eigenvectors[1][:, j]) for i, j in pairs])
    raw = np.array([lv[i] * rv[j] for i, j in pairs])
    ridged = np.array([(lv[i] / lv.mean() + 0.01) * (rv[j] / rv.mean() + 0.01) for i, j in pairs])
    info = dict(
        pairs=pairs,
        left_eigenvalues=lv.tolist(),
        right_eigenvalues=rv.tolist(),
        gram_resolution_rtol=rtol,
        left_excluded=len(lv) - len(li),
        right_excluded=len(rv) - len(ri),
        normalization_means=[float(lv.mean()), float(rv.mean())],
        ridge=0.01,
        rule="Distinct products nearest equally spaced log-product targets; stable index tie breaking",
    )
    return directions.astype(np.float32), np.diag(raw), np.diag(ridged), info


def block_functions(model, block_path, z_loss_weight):
    """Insert one flattened matrix slice while keeping every other parameter fixed."""
    pairs, treedef = jax.tree_util.tree_flatten_with_path(flatten_linear_layers(model))
    indices = [i for i, (path, _) in enumerate(pairs) if jax.tree_util.keystr(path) == block_path]
    if len(indices) != 1:
        raise ValueError(f"Ambiguous block path: {block_path}")
    index = indices[0]

    def loss(matrix, frozen_model, batch):
        leaves = jax.tree.leaves(flatten_linear_layers(frozen_model))
        bank = leaves[index]
        leaves[index] = bank.reshape((-1, *matrix.shape)).at[0].set(matrix).reshape(bank.shape)
        candidate = unflatten_linear_layers(frozen_model, treedef.unflatten(leaves))
        logits = candidate(batch.tokens, batch.attn_mask, key=None).astype(jnp.float32)
        return next_token_loss(
            candidate.Pos,
            candidate.Vocab,
            logits,
            batch.tokens,
            loss_weight=batch.loss_weight,
            logsumexp_weight=z_loss_weight,
        ).scalar()

    grad = jax.grad(loss, argnums=0)

    def hvp(matrix, direction, frozen_model, batch):
        return jax.jvp(lambda w: grad(w, frozen_model, batch), (matrix,), (direction,))[1]

    return loss, grad, hvp


def derivative_checks(matrix, directions, model, batch, loss, grad, hvp):
    """Validate low, middle and high Gram modes over a declared step-size sweep."""
    indices = sorted(set((0, len(directions) // 2, len(directions) - 1)))
    products = {i: host_array(hvp(matrix, jnp.asarray(directions[i]), model, batch)).astype(np.float64) for i in indices}
    first, last = indices[0], indices[-1]
    a = float(np.sum(directions[last] * products[first]))
    b = float(np.sum(directions[first] * products[last]))
    symmetry = abs(a - b) / max(np.linalg.norm(products[first]), np.linalg.norm(products[last]), 1e-30)
    checks = []
    for index in indices:
        e = jnp.asarray(directions[index])
        he = products[index]
        relative_errors = []
        for delta in (
            3.0,
            1.0,
            0.3,
            0.1,
            0.03,
            0.01,
            0.003,
            0.001,
            0.0003,
            0.0001,
            0.00003,
            0.00001,
            0.000003,
            0.000001,
        ):
            plus = host_array(grad(matrix + delta * e, model, batch)).astype(np.float64)
            minus = host_array(grad(matrix - delta * e, model, batch)).astype(np.float64)
            fd = (plus - minus) / (2 * delta)
            relative_errors.append(
                dict(
                    delta=delta,
                    relative_error=float(np.linalg.norm(fd - he) / max(np.linalg.norm(he), 1e-30)),
                    gradient_difference_norm=float(np.linalg.norm(plus - minus)),
                    gradient_norm=float(np.linalg.norm(plus)),
                )
            )
        valid = [r["relative_error"] < 0.02 for r in relative_errors]
        checks.append(
            dict(
                direction=index,
                hvp_norm=float(np.linalg.norm(he)),
                passed=any(x and y for x, y in pairwise(valid)),
                finite_difference=relative_errors,
            )
        )
    baseline = float(loss(matrix, model, batch))
    passed = np.isfinite(baseline) and symmetry < 1e-4 and all(c["passed"] for c in checks)
    return dict(
        passed=bool(passed),
        baseline_loss=baseline,
        symmetry_relative=float(symmetry),
        directions=checks,
        symmetry_tolerance=1e-4,
        fd_tolerance=0.02,
        requirement="Two adjacent step sizes below 2% relative HVP error in each tested mode",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-counter", type=int, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--directions", type=int, default=12)
    parser.add_argument("--logging-check", action="store_true")
    args = parser.parse_args()
    encoded = json.loads(args.configs.read_text())[0]["config"]["train_config"]
    cfg = resolve_config(encoded)
    if cfg.optimizer.alpha != 0.25 or cfg.optimizer.ridge != 0.01:
        raise ValueError("Expected the quarter-power reference config")
    identity = "ema-gram-hessian-" + uuid.uuid4().hex[:12]
    destination = Path(os.environ.get("IRIS_OUTPUT_DIR", "scratch/hessian")) / identity
    destination.mkdir(parents=True)
    run = wandb.init(
        entity="marin-community",
        project="marin",
        id=identity,
        name=identity,
        group="ema-gram-20-hessian",
        mode="online",
        resume="never",
        settings={"api_key": require_secret("WANDB_API_KEY")},
        config=dict(
            resolved_training_config=encoded,
            measurement_seed=args.seed,
            checkpoint=args.checkpoint,
            expected_counter=args.expected_counter,
            source_commit=os.environ.get("GIT_COMMIT"),
            groups=args.groups,
            directions=args.directions,
            logging_check=args.logging_check,
        ),
    )
    if run.settings.mode != "online":
        raise RuntimeError("Online tracking required")
    print(json.dumps(dict(event="tracking_started", url=run.url)), flush=True)
    run.log({"logging_check/success": 1}, step=0)
    if args.logging_check:
        run.finish()
        return
    initialize_jax()
    for key, value in cfg.trainer.jax_config.items():
        jax.config.update(key, value)
    jax.config.update("jax_default_matmul_precision", "highest")
    # Vanilla attention and unfused cross entropy provide differentiable FP32 paths.
    model_config = replace(cfg.model, attn_backend=AttentionBackend.VANILLA)
    bucket_name, prefix = args.output_prefix.removeprefix("gs://").rstrip("/").split("/", 1)
    bucket = storage.Client().bucket(bucket_name)
    prefix += "/" + identity
    log_step = 0

    def save(name, payload):
        path = destination / name
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        if jax.process_index() == 0:
            bucket.blob(prefix + "/" + name).upload_from_filename(str(path))

    def progress(payload):
        nonlocal log_step
        log_step += 1
        payload = dict(step=log_step, **payload)
        with (destination / "measurements.jsonl").open("a") as stream:
            stream.write(json.dumps(payload, allow_nan=False) + "\n")
        run.log(payload, step=log_step)
        print(json.dumps(payload), flush=True)
        if jax.process_index() == 0:
            bucket.blob(prefix + "/measurements.jsonl").upload_from_filename(str(destination / "measurements.jsonl"))

    try:
        with cfg.trainer.use_device_mesh():
            vocab = hax.partitioning.round_axis_for_partitioning(
                hax.Axis("vocab", len(cfg.data.the_tokenizer)), cfg.trainer.compute_axis_mapping
            )
            tx = cfg.optimizer.build(cfg.trainer.num_train_steps)
            template = eqx.filter_eval_shape(
                lambda: TrainerState.init(
                    tx,
                    model_config.build(vocab, key=jax.random.PRNGKey(0)),
                    key=jax.random.PRNGKey(0),
                    mp=cfg.trainer.mp,
                )
            )
            start = time.perf_counter()
            state = load_checkpoint(
                template, args.checkpoint, axis_mapping=cfg.trainer.parameter_axis_mapping, allow_partial=False
            )
            if int(state.step) != args.expected_counter:
                raise ValueError("Full restore counter mismatch")
            model = inference_mode(state.model, True)
            if any(x.dtype != jnp.float32 for x in jax.tree.leaves(model) if eqx.is_inexact_array(x)):
                raise ValueError("All floating model parameters must be FP32")
            progress(dict(event="restored", counter=int(state.step), seconds=time.perf_counter() - start))
            dataset = cfg.data.train_set(
                cfg.model.max_Pos, cfg.trainer.batch_schedule, key=jax.random.PRNGKey(args.seed)
            )
            # Each group uses the native weighted-token mean; groups receive equal weight.
            batch_axis = hax.Axis("batch", cfg.trainer.train_batch_size)
            batches = list(
                islice(
                    DataLoader(dataset, batch_size=batch_axis, axis_resources=cfg.trainer.compute_axis_mapping),
                    args.groups,
                )
            )
            if len(batches) != args.groups:
                raise ValueError("Insufficient fresh data groups")
            seen = set()
            manifest = []
            for bi, batch in enumerate(batches):
                tokens = host_array(batch.tokens.array)
                weights = host_array(batch.loss_weight.array)
                ids = [hashlib.sha256(x.tobytes()).hexdigest() for x in tokens]
                if any(x in seen for x in ids) or len(set(ids)) != len(ids):
                    raise ValueError("Duplicate token sequences in curvature groups")
                seen.update(ids)
                manifest.append(
                    dict(
                        group=bi,
                        mixture_indices=list(range(bi * len(ids), (bi + 1) * len(ids))),
                        token_sha256=ids,
                        tokens=int(tokens.size),
                        valid_weight=float(weights[..., :-1].sum()),
                    )
                )
            save(
                "data-manifest.json",
                dict(
                    groups=manifest,
                    seed=args.seed,
                    data_config=encoded["data"],
                    weighting="Equal group weights; native valid-weight-normalized token mean within each group",
                    scope="Fresh permutation of fixed training caches; not claimed unseen during training",
                ),
            )
            masked, _, optimizer = matrix_state(model, state.opt_state)
            paths = jax.tree_util.tree_flatten_with_path(flatten_linear_layers(masked))[0]
            full_paths = dict(
                (jax.tree_util.keystr(p), v)
                for p, v in jax.tree_util.tree_flatten_with_path(flatten_linear_layers(model))[0]
            )
            selected = [
                (jax.tree_util.keystr(p), s)
                for (p, _), s in zip(paths, optimizer.matrices, strict=True)
                if any(name in jax.tree_util.keystr(p) for name in ("self_attn.q_proj.weight", "mlp.gate_proj.weight"))
            ]
            if len(selected) != 2:
                raise ValueError(f"Expected q_proj and gate_proj, found {[p for p, _ in selected]}")
            for block, gram_state in selected:
                label = "attention" if "q_proj" in block else "mlp"
                matrix = full_paths[block].reshape((-1, *gram_state.momentum.shape[-2:]))[0]
                left = host_array(gram_state.left.reshape((-1, *gram_state.left.shape[-2:]))[0])
                right = host_array(gram_state.right.reshape((-1, *gram_state.right.shape[-2:]))[0])
                directions, raw, ridged, selection = select_basis(left, right, args.directions)
                save(label + "-basis.json", dict(block=block, flat_index=0, shape=list(matrix.shape), **selection))
                basis_file = destination / (label + "-basis.npz")
                np.savez_compressed(basis_file, directions=directions, gram_raw=raw, gram_ridged=ridged)
                if jax.process_index() == 0:
                    bucket.blob(prefix + "/" + basis_file.name).upload_from_filename(str(basis_file))
                loss, grad, hvp = [
                    hax.named_jit(fn, axis_resources=cfg.trainer.compute_axis_mapping)
                    for fn in block_functions(model, block, cfg.z_loss_weight)
                ]
                start = time.perf_counter()
                checks = derivative_checks(matrix, directions, model, batches[0], loss, grad, hvp)
                checks["seconds_including_compile"] = time.perf_counter() - start
                save(label + "-derivative-checks.json", checks)
                if not checks["passed"]:
                    raise ValueError(f"Derivative validation failed for {block}: {checks}")
                progress(
                    dict(
                        event="derivatives_validated",
                        block=label,
                        loss=checks["baseline_loss"],
                        seconds=checks["seconds_including_compile"],
                    )
                )
                d = len(directions)
                curvature = np.zeros((args.groups, d, d), np.float64)
                leakage = np.zeros((args.groups, d), np.float64)
                total_norm = 0.0
                residual_norm = 0.0
                durations = []
                for j, direction in enumerate(directions):
                    mean_hvp = np.zeros_like(direction, dtype=np.float64)
                    for bi, batch in enumerate(batches):
                        start = time.perf_counter()
                        value = host_array(hvp(matrix, jnp.asarray(direction), model, batch)).astype(np.float64)
                        elapsed = time.perf_counter() - start
                        if not np.isfinite(value).all():
                            raise ValueError("Nonfinite HVP")
                        projection = np.einsum("iab,ab->i", directions, value)
                        curvature[bi, :, j] = projection
                        norm2 = np.sum(value * value)
                        leakage[bi, j] = np.sqrt(max(0.0, norm2 - np.sum(projection**2)) / max(norm2, 1e-300))
                        mean_hvp += value / args.groups
                        durations.append(elapsed)
                        progress(
                            dict(
                                event="hvp",
                                block=label,
                                direction=j,
                                group=bi,
                                seconds=elapsed,
                                projected_norm=float(np.linalg.norm(projection)),
                            )
                        )
                    projection = np.einsum("iab,ab->i", directions, mean_hvp)
                    norm2 = np.sum(mean_hvp**2)
                    total_norm += norm2
                    residual_norm += max(0.0, norm2 - np.sum(projection**2))
                    save(label + "-partial.json", dict(completed_directions=j + 1, curvature_batches=curvature.tolist()))
                metadata = dict(
                    curvature_kind="hessian",
                    checkpoint=args.checkpoint,
                    block=block,
                    trajectory="alpha0.25_seed0",
                    precision="FP32 parameters, forward, derivatives; highest matmul; FP64 eigendecomposition",
                    curvature_shift=0.0,
                    source_commit=os.environ.get("GIT_COMMIT"),
                    counter=int(state.step),
                    flat_index=0,
                    loss="next-token cross entropy plus configured z-loss; vanilla attention; unfused loss",
                    z_loss_weight=cfg.z_loss_weight,
                    weight_decay_included=False,
                    gradient_history="global reduced/clipped gradients; max norm 0.1",
                    data_manifest="data-manifest.json",
                    measurement_seed=args.seed,
                    selection=selection,
                    derivative_checks=checks,
                    hvp_seconds=durations,
                    measurement_resolution="Assess derivative/batch stability; 1e-12 is only a numerical gate",
                )
                path = destination / (label + ".npz")
                np.savez_compressed(
                    path,
                    curvature_batches=curvature,
                    gram_raw=raw,
                    gram_ridged=ridged,
                    leakage=leakage,
                    leakage_mean_hvp=np.sqrt(residual_norm / max(total_norm, 1e-300)),
                    metadata_json=json.dumps(metadata),
                )
                if jax.process_index() == 0:
                    bucket.blob(prefix + "/" + path.name).upload_from_filename(str(path))
                progress(
                    dict(
                        event="block_complete",
                        block=label,
                        output="gs://" + bucket_name + "/" + prefix + "/" + path.name,
                    )
                )
            save(
                "completed.json",
                dict(success=True, counter=int(state.step), output="gs://" + bucket_name + "/" + prefix, wandb=run.url),
            )
    except BaseException as error:
        save("failure.json", dict(error=repr(error), wandb=run.url))
        run.finish(exit_code=1)
        raise
    else:
        run.finish()


if __name__ == "__main__":
    main()
