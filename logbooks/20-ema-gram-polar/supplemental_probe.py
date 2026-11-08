# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fresh-batch local loss controls from unchanged regional TrainerState checkpoints."""

import hashlib
import time
from functools import partial
from itertools import islice

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from levanter.checkpoint import load_checkpoint
from levanter.data.loader import DataLoader
from levanter.optim.ema_gram import NEWTON_SCHULZ, SVD, gram_direction
from levanter.optim.util import unflatten_linear_layers
from levanter.trainer_state import TrainerState
from levanter.utils.tree_utils import inference_mode

from experiments.ema_gram.probe import matrix_state, resolve_config


@partial(jax.jit, static_argnames=("alpha", "ridge", "method", "polar_steps"))
def matched_bank(bank, alpha, ridge, method, polar_steps):
    reference = gram_direction(bank, alpha=0.0, ridge=ridge, polar_steps=polar_steps)
    candidate = gram_direction(bank, alpha=alpha, ridge=ridge, polar_steps=polar_steps, polar_method=method)
    target = jnp.linalg.norm(reference, axis=(-2, -1), keepdims=True)
    actual = jnp.linalg.norm(candidate, axis=(-2, -1), keepdims=True)
    return candidate * target / jnp.where(actual > 0, actual, 1)


def matched_direction(model, opt_state, alpha, ridge, method, polar_steps):
    """Match each matrix, bounding decomposition temporaries to one stored bank."""
    masked, treedef, state = matrix_state(model, opt_state)
    directions = []
    for bank in state.matrices:
        result = matched_bank(bank, alpha, ridge, method, polar_steps)
        jax.block_until_ready(result)
        directions.append(result)
    direction = unflatten_linear_layers(masked, treedef.unflatten(directions))
    return jax.tree.map(
        lambda d, p: jax.tree.map(jnp.zeros_like, p) if isinstance(d, optax.MaskedNode) else d,
        direction,
        model,
        is_leaf=lambda x: isinstance(x, optax.MaskedNode),
    )


@partial(jax.jit, static_argnames=("alpha", "ridge", "polar_steps"))
def production_bank_directions(banks, alpha, ridge, polar_steps):
    """Compute matrix directions only; exclude gradient, Gram updates and norm matching."""
    return tuple(gram_direction(b, alpha=alpha, ridge=ridge, polar_steps=polar_steps) for b in banks)


def direction_timings(banks, ridge, polar_steps):
    """Record warm, synchronized kernel timings in forward and reversed alpha order."""
    alphas = (0.0, 0.25, 0.5)
    for alpha in alphas:
        jax.block_until_ready(production_bank_directions(banks, alpha, ridge, polar_steps))
    rows = []
    for repeat in range(4):
        for alpha in alphas if repeat % 2 == 0 else alphas[::-1]:
            start = time.perf_counter()
            outputs = production_bank_directions(banks, alpha, ridge, polar_steps)
            jax.block_until_ready(outputs)
            rows.append(dict(alpha=alpha, repeat=repeat, seconds=time.perf_counter() - start))
            del outputs
    return rows


def summarize_losses(losses, baseline):
    delta = np.asarray(losses) - np.asarray(baseline)
    if not np.isfinite(delta).all() or not np.isfinite(losses).all():
        raise ValueError("Nonfinite paired loss")
    return dict(
        delta_mean=float(delta.mean()),
        delta_std=float(delta.std(ddof=1)),
        delta_sem=float(delta.std(ddof=1) / np.sqrt(len(delta))),
    )


def select_batches(loader, *, offset, count, available_sequences, batch_size):
    """Reject incomplete ranges before loading and seek directly to the selected batch."""
    full_batches = available_sequences // batch_size
    if offset < 0 or count < 2 or offset + count > full_batches:
        raise ValueError(f"Requested batches [{offset}, {offset + count}) exceed {full_batches} full batches")
    batches = list(islice(loader.iter_from_step(offset), count))
    if len(batches) != count:
        raise ValueError("Dataset ended before its declared length")
    return batches


def run(settings, encoded_config, emit):
    """Restore three states in one process, reusing loss and direction compilation."""
    cfg = resolve_config(encoded_config)
    for key, value in cfg.trainer.jax_config.items():
        jax.config.update(key, value)
    with cfg.trainer.use_device_mesh():
        vocab = hax.partitioning.round_axis_for_partitioning(
            hax.Axis("vocab", len(cfg.data.the_tokenizer)), cfg.trainer.compute_axis_mapping
        )
        tx = cfg.optimizer.build(cfg.trainer.num_train_steps)
        template = eqx.filter_eval_shape(
            lambda: TrainerState.init(
                tx, cfg.model.build(vocab, key=jax.random.PRNGKey(0)), key=jax.random.PRNGKey(0), mp=cfg.trainer.mp
            )
        )
        dataset = cfg.data.validation_sets(cfg.model.max_Pos)[settings["dataset"]]
        loader = DataLoader(dataset, batch_size=cfg.trainer.EvalBatch, axis_resources=cfg.trainer.compute_axis_mapping)
        offset, count = settings["batch_offset"], settings["batches"]
        available_sequences = len(dataset.as_sync_dataset())
        batches = select_batches(
            loader,
            offset=offset,
            count=count,
            available_sequences=available_sequences,
            batch_size=cfg.trainer.eval_batch_size,
        )
        # Record per-rank local shard fingerprints without transferring model arrays.
        fingerprints = []
        for batch in batches:
            digest = hashlib.sha256()
            for leaf in jax.tree.leaves(batch):
                for shard in leaf.addressable_shards:
                    digest.update(str(shard.index).encode())
                    digest.update(np.asarray(shard.data).tobytes())
            fingerprints.append(digest.hexdigest())
        emit(
            dict(
                kind="data",
                available_sequences=available_sequences,
                batch_offset=offset,
                batches=count,
                local_shard_sha256=fingerprints,
                process_index=jax.process_index(),
                process_count=jax.process_count(),
                device_count=jax.device_count(),
                eval_batch_size=cfg.trainer.eval_batch_size,
            )
        )

        @hax.named_jit(axis_resources=cfg.trainer.compute_axis_mapping)
        def loss_fn(model, batch):
            model = cfg.trainer.mp.cast_to_compute(inference_mode(model, True))
            return model.compute_next_token_loss(batch, key=None).scalar()

        for request in settings["checkpoints"]:
            state = load_checkpoint(
                template, request["checkpoint"], axis_mapping=cfg.trainer.parameter_axis_mapping, allow_partial=False
            )
            counter = int(state.step)
            if counter != request["counter"]:
                raise ValueError("Restored counter differs from bound reference")
            emit(dict(kind="restore", counter=counter, checkpoint=request["checkpoint"], allow_partial=False))
            baseline = [float(loss_fn(state.model, b)) for b in batches]
            methods = [(a, NEWTON_SCHULZ) for a in settings["alphas"]]
            if counter == settings["svd_counter"]:
                methods.append((0.25, SVD))
            for alpha, method in methods:
                start = time.perf_counter()
                direction = matched_direction(
                    state.model, state.opt_state, float(alpha), settings["ridge"], method, cfg.optimizer.polar_steps
                )
                jax.block_until_ready(direction)
                emit(
                    dict(
                        kind="direction",
                        counter=counter,
                        alpha=alpha,
                        method=method,
                        compile_inclusive_seconds=time.perf_counter() - start,
                    )
                )
                for size in settings["step_sizes"]:
                    candidate = eqx.apply_updates(state.model, jax.tree.map(lambda d, size=size: -size * d, direction))
                    losses = [float(loss_fn(candidate, b)) for b in batches]
                    emit(
                        dict(
                            kind="loss",
                            counter=counter,
                            alpha=alpha,
                            method=method,
                            ridge=settings["ridge"],
                            step_size=size,
                            baseline_losses=baseline,
                            losses=losses,
                            **summarize_losses(losses, baseline),
                        )
                    )
                    del candidate
                del direction
            if counter == settings["svd_counter"]:
                banks = matrix_state(state.model, state.opt_state).optimizer.matrices
                emit(
                    dict(
                        kind="matrix_direction_timings",
                        counter=counter,
                        scope="Same stored banks; Gram powers plus production polar only; no training-step claim",
                        rows=direction_timings(banks, settings["ridge"], cfg.optimizer.polar_steps),
                        state_bytes={
                            field: sum(getattr(b, field).size * getattr(b, field).dtype.itemsize for b in banks)
                            for field in ("momentum", "left", "right")
                        },
                    )
                )
                del banks
            emit(
                dict(
                    kind="memory",
                    counter=counter,
                    scope="Cumulative probe process; not comparative training peak",
                    devices=[d.memory_stats() for d in jax.local_devices()],
                )
            )
            del state
