# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.models.qwen import QwenConfig
from levanter.trainer_state import TrainerState

from experiments.ema_gram.muon_checkpoint import july_template, migrate_july_state, restore_muon_checkpoint
from experiments.speedrun.prism_berkeley_qwen3_scaling.muon_error_feedback_optimizer import ErrorAwareMuonConfig


def test_july_schema_restores_all_arrays_and_next_update_without_partial_loading(tmp_path):
    cfg = QwenConfig(hidden_dim=8, intermediate_dim=16, num_layers=1, num_heads=2, num_kv_heads=2, max_seq_len=8)
    optimizer = ErrorAwareMuonConfig(
        learning_rate=0.012, adam_lr=0.0036, momentum=0.98, policy="muon", correction_warmup_steps=0, warmup=0
    ).build(100)
    model = cfg.build(hax.Axis("vocab", 16), key=jax.random.PRNGKey(12))
    state = TrainerState.init(
        optimizer, model, key=jax.random.PRNGKey(54), mp=jmp.get_policy("params=float32,compute=float32,output=float32")
    )
    rng = np.random.default_rng(31)
    gradient = jax.tree.map(lambda x: jnp.asarray(rng.normal(size=x.shape), x.dtype) * 0.01, model)
    update = hax.named_jit(optimizer.update, axis_resources={})
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(-1, 1), ("data", "model"))
    with jax.sharding.set_mesh(mesh):
        for _ in range(2):
            _, opt_state = update(gradient, state.opt_state, model)
            state = eqx.tree_at(lambda x: (x.step, x.opt_state), state, (state.step + 1, opt_state))
        legacy = july_template(state)
        path = str(tmp_path / "legacy")
        save_checkpoint(legacy, 2, path)
        with pytest.raises(FileNotFoundError, match="count"):
            load_checkpoint(state, path, axis_mapping={}, allow_partial=False)
        restored, record = restore_muon_checkpoint(
            state, dict(checkpoint=path, expected_counter=2, checkpoint_schema="momentum_only_7cd118640"), {}
        )
        for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
            np.testing.assert_array_equal(a, b)
        expected, next_state = update(gradient, state.opt_state, model)
        actual, migrated_next = update(gradient, restored.opt_state, model)
        for a, b in zip(jax.tree.leaves((expected, next_state)), jax.tree.leaves((actual, migrated_next)), strict=True):
            np.testing.assert_array_equal(a, b)
        assert record["saved_counters"]["adam"] == 2
        corrupt = eqx.tree_at(lambda x: x.step, legacy, legacy.step + 1)
        with pytest.raises(ValueError, match="inconsistent saved counters"):
            migrate_july_state(corrupt, 2)
