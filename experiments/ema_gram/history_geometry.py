# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-matrix frozen-history controls, keeping nonmatrix optimizer state intact."""

import hashlib
import json

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from levanter.data.loader import DataLoader
from levanter.optim.ema_gram import EmaGramState, GramState, update_gram_state
from levanter.optim.util import flatten_linear_layers

from experiments.ema_gram.hessian import host_array
from experiments.ema_gram.probe import matrix_state


def replace_matrix_history(opt_state, replacement):
    count = sum(
        isinstance(x, EmaGramState) for x in jax.tree.leaves(opt_state, is_leaf=lambda x: isinstance(x, EmaGramState))
    )
    if count != 1:
        raise ValueError("Expected one complete matrix-history state")
    return jax.tree.map(
        lambda x: replacement if isinstance(x, EmaGramState) else x,
        opt_state,
        is_leaf=lambda x: isinstance(x, EmaGramState),
    )


def combine_histories(saved, fresh, *, momentum, grams):
    if momentum not in ("saved", "fresh") or grams not in ("saved", "fresh"):
        raise ValueError("History sources must be saved or fresh")
    return EmaGramState(
        tuple(
            GramState(
                s.momentum if momentum == "saved" else f.momentum,
                s.left if grams == "saved" else f.left,
                s.right if grams == "saved" else f.right,
            )
            for s, f in zip(saved.matrices, fresh.matrices, strict=True)
        )
    )


def pool_raw_histories(histories):
    """Average raw EMA buffers before any trace normalization, ridge or powers."""
    if len(histories) != 2:
        raise ValueError("The prespecified pool has two independent histories")
    return jax.tree.map(
        lambda a, b: ((np.asarray(a, np.float64) + np.asarray(b, np.float64)) / 2).astype(np.float32), *histories
    )


def frozen_history_step(cfg, native_model, opt_state):
    masked, _, saved = matrix_state(native_model, opt_state)

    def step(gradient, history):
        clipped, _ = optax.clip_by_global_norm(cfg.optimizer.max_grad_norm).update(gradient, ())
        filtered = jax.tree.map(
            lambda p, g: optax.MaskedNode() if isinstance(p, optax.MaskedNode) else g,
            masked,
            clipped,
            is_leaf=lambda x: isinstance(x, optax.MaskedNode),
        )
        gradients = jax.tree.leaves(flatten_linear_layers(filtered))
        return EmaGramState(
            tuple(
                update_gram_state(g, s, cfg.optimizer.momentum, cfg.optimizer.gram_beta)
                for g, s in zip(gradients, history.matrices, strict=True)
            )
        )

    compiled = hax.named_jit(step, axis_resources=cfg.trainer.compute_axis_mapping)
    zero = jax.tree.map(jnp.zeros_like, saved)
    return compiled, zero


def collect_frozen_history(
    *,
    cfg,
    native_model,
    opt_state,
    gradient_function,
    seed,
    count,
    excluded_hashes,
    save,
    save_arrays,
    progress,
    replica,
):
    """Collect one zero-initialized history on text disjoint from every other role."""
    if count != 512:
        raise ValueError("The full checkpoint protocol requires512 fresh batches per replica")
    seen = set(excluded_hashes)
    update_history, history = frozen_history_step(cfg, native_model, opt_state)
    dataset = cfg.data.train_set(cfg.model.max_Pos, cfg.trainer.batch_schedule, key=jax.random.PRNGKey(seed))
    loader = DataLoader(
        dataset,
        batch_size=hax.Axis("batch", cfg.trainer.train_batch_size),
        axis_resources=cfg.trainer.compute_axis_mapping,
    )
    records, rejected = [], []
    for source_index, batch in enumerate(loader):
        hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in host_array(batch.tokens.array)]
        if len(set(hashes)) != len(hashes) or seen.intersection(hashes):
            rejected.append(source_index)
            continue
        seen.update(hashes)
        loss, gradient = gradient_function(native_model, batch)
        norm = float(optax.global_norm(gradient))
        history = update_history(gradient, history)
        jax.block_until_ready(history)
        if not np.isfinite([float(loss), norm]).all():
            raise ValueError("Nonfinite frozen-history gradient")
        records.append(
            dict(
                index=len(records),
                source_index=source_index,
                sequence_hashes=hashes,
                loss=float(loss),
                gradient_norm=norm,
                clipping_factor=min(1.0, cfg.optimizer.max_grad_norm / max(norm, 1e-30)),
            )
        )
        del gradient
        if len(records) % 32 == 0:
            save(f"history-{replica}-manifest.json", dict(seed=seed, records=records, rejected=rejected))
            progress(dict(event="history_progress", replica=replica, batches=len(records), loss=float(loss)))
        if len(records) == count:
            break
    if len(records) != count:
        raise ValueError("History text exhausted before512 disjoint batches")
    host_history = jax.tree.map(host_array, history)
    arrays = {
        f"matrix_{i}_{name}": getattr(s, name)
        for i, s in enumerate(host_history.matrices)
        for name in ("momentum", "left", "right")
    }
    artifact = save_arrays(f"history-{replica}-raw.npz", **arrays)
    manifest = dict(
        seed=seed,
        records=records,
        rejected=rejected,
        arrays=artifact,
        initialization="All raw matrix buffers zero; nonmatrix state and weights unchanged",
        zero_initialization_mass=dict(momentum=cfg.optimizer.momentum**count, grams=cfg.optimizer.gram_beta**count),
        scope="Fixed weights and original clipped production gradients on disjoint history text",
    )
    manifest["records_sha256"] = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    save(f"history-{replica}-completed.json", manifest)
    return host_history, seen, manifest
