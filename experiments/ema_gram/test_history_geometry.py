# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
from levanter.layers.attention import AttentionBackend
from levanter.models.lm_model import LmExample
from levanter.models.qwen import QwenConfig
from levanter.optim.ema_gram import EmaGramConfig, EmaGramState

from experiments.ema_gram import history_comparison
from experiments.ema_gram.history_geometry import (
    combine_histories,
    frozen_history_step,
    pool_raw_histories,
    replace_matrix_history,
)
from experiments.ema_gram.probe import matrix_state


def test_all_matrix_histories_match_production_and_leave_nonmatrix_state_untouched():
    config = QwenConfig(hidden_dim=8, intermediate_dim=16, num_layers=2, num_heads=2, num_kv_heads=2, max_seq_len=8)
    model = config.build(hax.Axis("vocab", 32), key=jax.random.PRNGKey(17))
    optimizer_config = EmaGramConfig(learning_rate=0.001, warmup=0, alpha=0.125, ridge=0.01)
    optimizer = optimizer_config.build(100)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    cfg = SimpleNamespace(optimizer=optimizer_config, trainer=SimpleNamespace(compute_axis_mapping={}))
    gradient = jax.tree.map(lambda x: jnp.ones_like(x) * 0.03 if eqx.is_array(x) else x, model)
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(-1, 1), ("data", "model"))
    with jax.sharding.set_mesh(mesh):
        advance, history = frozen_history_step(cfg, model, state)
        history = advance(gradient, history)
        _, advanced = optimizer.update(gradient, state, eqx.filter(model, eqx.is_inexact_array))
    production = matrix_state(model, advanced).optimizer
    for actual, expected in zip(jax.tree.leaves(history), jax.tree.leaves(production), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
    fresh = jax.tree.map(lambda x: x * 2, history)
    mixed = combine_histories(production, fresh, momentum="fresh", grams="saved")
    replaced = replace_matrix_history(advanced, mixed)

    def without_matrix(x):
        return isinstance(x, EmaGramState)

    for before, after in zip(
        jax.tree.leaves(advanced, is_leaf=without_matrix), jax.tree.leaves(replaced, is_leaf=without_matrix), strict=True
    ):
        if not isinstance(before, EmaGramState):
            np.testing.assert_array_equal(before, after)
    for saved, candidate in zip(production.matrices, mixed.matrices, strict=True):
        np.testing.assert_allclose(candidate.momentum, 2 * saved.momentum)
        np.testing.assert_array_equal(candidate.left, saved.left)
        np.testing.assert_array_equal(candidate.right, saved.right)
    pooled = pool_raw_histories([history, fresh])
    for actual, raw in zip(jax.tree.leaves(pooled), jax.tree.leaves(history), strict=True):
        np.testing.assert_allclose(actual, 1.5 * raw, rtol=1e-6)
    assert float(optax.global_norm(history)) > 0


def test_history_pipeline_pairs_conditions_and_locks_before_testing(tmp_path, monkeypatch):
    config = QwenConfig(
        hidden_dim=8,
        intermediate_dim=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=8,
        attn_backend=AttentionBackend.VANILLA,
    )
    model = config.build(hax.Axis("vocab", 16), key=jax.random.PRNGKey(2))
    optimizer = EmaGramConfig(learning_rate=0.001, warmup=0, alpha=0.125, ridge=0.01)
    cfg = SimpleNamespace(
        optimizer=optimizer,
        trainer=SimpleNamespace(
            compute_axis_mapping={},
            num_train_steps=100,
            mp=jmp.get_policy("params=float32,compute=float32,output=float32"),
        ),
        z_loss_weight=1e-7,
    )
    opt_state = optimizer.build(100).init(eqx.filter(model, eqx.is_inexact_array))
    batch = LmExample.causal(hax.named(jnp.array([1, 4, 2, 3, 6, 2, 1, 0]), config.max_Pos))
    saved, events, seen_inputs = {}, [], []
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text('["earlier-study"]')

    def save(name, value):
        saved[name] = json.loads(json.dumps(value, allow_nan=False))

    def collect(**kwargs):
        seen_inputs.append(set(kwargs["excluded_hashes"]))
        advance, zero = frozen_history_step(cfg, model, opt_state)
        rng = np.random.default_rng(kwargs["seed"])
        gradient = jax.tree.map(lambda x: jnp.asarray(rng.normal(size=x.shape), x.dtype) * 0.02, model)
        fresh = advance(gradient, zero)
        seen = set(kwargs["excluded_hashes"]) | {f"history-{kwargs['replica']}"}
        return jax.tree.map(np.asarray, fresh), seen, dict(replica=kwargs["replica"], test_double=True)

    monkeypatch.setattr(history_comparison, "collect_frozen_history", collect)
    protocol = dict(
        history=dict(
            seeds=[91, 92],
            batches_per_replica=512,
            combinations=[["saved", "saved"], ["fresh", "saved"], ["saved", "fresh"], ["fresh", "fresh"]],
        ),
        length_grid=[0, 0.1, 0.3, 0.6, 1, 2, 4],
    )
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(-1, 1), ("data", "model"))
    with jax.sharding.set_mesh(mesh):
        native_grad, functions = history_comparison.production_functions(cfg, alphas=(0.125,), modes=("full",))
        _, g = native_grad(model, batch)
        _, opt_state = functions["full", 0.125](g, opt_state, model)
        result = history_comparison.run_history_comparison(
            state=SimpleNamespace(step=51, opt_state=opt_state),
            model=model,
            native_model=model,
            cfg=cfg,
            panels=dict(update=[batch], calibration=[batch] * 8, evaluation=[batch]),
            weights=dict(update=[7], calibration=[7] * 8, evaluation=[7]),
            args=SimpleNamespace(state="tiny", exclusions=exclusions),
            protocol=protocol,
            save=save,
            progress=events.append,
            manifest_hash="test",
            save_arrays=None,
        )
    assert "earlier-study" in seen_inputs[0] and "history-0" in seen_inputs[1]
    assert len(result["rows"]) == 20
    assert all("evaluation" not in row for row in saved["locked-predictions.json"]["rows"])
    assert next(i for i, e in enumerate(events) if e["event"] == "predictions_locked") < next(
        i for i, e in enumerate(events) if e["event"] == "history_heldout"
    )
    assert len({row["evaluation"]["equal"]["length"] for row in result["rows"]}) == 1
    assert {row["condition"] for row in result["rows"]} == {c["name"] for c in result["conditions"]}
