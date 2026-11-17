# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel

from experiments.evaluation.inference_correctness.reference import score_snowball_positions


def test_selected_reference_scores_match_independent_full_logits():
    config = SnowballConfig(
        vocab_size=24,
        hidden_dim=12,
        intermediate_dim=16,
        shared_expert_intermediate_dim=12,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=16,
        sliding_window=4,
        attention_implementation="reference",
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", 24), config, key=jax.random.key(7))
        tokens = jnp.asarray([[3, 8, 9, 4, 7, 2, 1, 5]], dtype=jnp.int32)
        named = hax.named(tokens, (Axis("batch", 1), Axis("position", 8)))
        values, token_ids = hax.named_jit(score_snowball_positions)(model, named, jnp.asarray([[0, 3, 7]]))
        logits = hax.named_jit(lambda m, ids: m(ids))(model, named).array
        expected = np.asarray(jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1))
        values, token_ids = np.asarray(values), np.asarray(token_ids)
    for column, position in enumerate([0, 3, 7]):
        np.testing.assert_allclose(values[0, column], expected[0, position, token_ids[0, column]], atol=1e-6, rtol=1e-6)
        assert token_ids[0, column, 0] == expected[0, position].argmax()
