# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise frozen-control data separation and full optimizer branching end to end."""

from types import SimpleNamespace

import equinox as eqx
import haliax as hax
import jax
import jmp
import numpy as np
from levanter.models.lm_model import LmExample
from levanter.models.qwen import QwenConfig
from levanter.optim.ema_gram import EmaGramConfig
from levanter.optim.util import flatten_linear_layers

from experiments.ema_gram import frozen_controls
from experiments.ema_gram.hessian import select_basis


def test_frozen_controls_preserve_checkpoint_and_pair_data(monkeypatch):
    model_config = QwenConfig(
        hidden_dim=8, intermediate_dim=16, num_layers=1, num_heads=2, num_kv_heads=2, max_seq_len=8
    )
    model = model_config.build(hax.Axis("vocab", 32), key=jax.random.PRNGKey(7))
    optimizer = EmaGramConfig(learning_rate=0.001, warmup=0.0, alpha=0.125, ridge=0.01)
    opt_state = optimizer.build(100).init(eqx.filter(model, eqx.is_inexact_array))
    state = SimpleNamespace(step=1, opt_state=opt_state)
    before = [np.asarray(x).copy() for x in jax.tree.leaves((model, opt_state)) if eqx.is_array(x)]

    def data(pos, schedule, *, key):
        batches = []
        keys = [jax.random.PRNGKey(999), *jax.random.split(key, 20)]
        for k in keys:
            tokens = jax.random.randint(k, (1, 8), 0, 32)
            batch_axis = hax.Axis("batch", 1)
            batches.append(hax.vmap(LmExample.causal, batch_axis)(hax.named(tokens, (batch_axis, pos))))
        return batches

    cfg = SimpleNamespace(
        optimizer=optimizer,
        trainer=SimpleNamespace(
            mp=jmp.get_policy("params=float32,compute=float32,output=float32"),
            compute_axis_mapping={},
            train_batch_size=1,
            batch_schedule=None,
            num_train_steps=100,
        ),
        data=SimpleNamespace(train_set=data),
        model=model_config,
        z_loss_weight=1e-7,
    )
    monkeypatch.setattr(frozen_controls, "DataLoader", lambda dataset, **kwargs: iter(dataset))
    blocks = []
    for path, value in jax.tree_util.tree_flatten_with_path(flatten_linear_layers(model))[0]:
        name = jax.tree_util.keystr(path)
        if "q_proj.weight" in name or "gate_proj.weight" in name:
            shape = value.shape[-2:]
            directions, _, _, _ = select_basis(np.eye(shape[0]), np.eye(shape[1]), 4)
            blocks.append(dict(label=name, path=name, shape=shape, directions=directions))
    outputs = {}
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(-1, 1), ("data", "model"))
    with jax.sharding.set_mesh(mesh):
        frozen_controls.run_controls(
            state, model, cfg, blocks, outputs.__setitem__, lambda x: None, np.asarray, history_batches=2
        )
    assert outputs["controls-completed.json"]["update_candidates"] == 30
    assert len(outputs["fresh-grams.json"]["results"]) == 4
    manifest = outputs["controls-data-manifest.json"]
    assert len(manifest) == 8 + 4 + 17
    assert len(outputs["controls-rejected-batches.json"]) == 3
    rows = outputs["full-update-losses.json"]["rows"]
    assert all(row["baseline_losses"] == rows[0]["baseline_losses"] for row in rows)
    for fraction in (0.1, 0.3, 1.0):
        matched = [
            r["applied_update_norm"] for r in rows if r["matching"] != "native" and r["step_fraction"] == fraction
        ]
        np.testing.assert_allclose(matched, matched[0], rtol=1e-6)
    after = [np.asarray(x) for x in jax.tree.leaves((model, opt_state)) if eqx.is_array(x)]
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(a, b)
    assert state.step == 1
