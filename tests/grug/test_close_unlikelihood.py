# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug_sft.special_token_train import CloseUnlikelihoodConfig, post_close_positions
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, TokenUnlikelihood, Transformer

BOT, OPEN, CLOSE, HDR, EOT = 1, 2, 3, 4, 5
CONFIG = CloseUnlikelihoodConfig(
    close_token_id=CLOSE, open_token_id=OPEN, turn_boundary_token_ids=(BOT, HDR, EOT), weight=2.0
)
MODEL = GrugModelConfig(
    vocab_size=16,
    hidden_dim=8,
    intermediate_dim=8,
    shared_expert_intermediate_dim=8,
    num_experts=2,
    num_experts_per_token=1,
    num_layers=2,
    num_heads=2,
    num_kv_heads=2,
    max_seq_len=16,
    sliding_window=8,
    attention_implementation="reference",
    moe_implementation="scatter",
    disable_pko=True,
    disable_long_rope=True,
    use_array_stacked_blocks=True,
)


def test_post_close_positions_cover_a_closed_block_until_it_reopens_or_the_turn_ends():
    # Turn 1: open, reason, close, answer, eot. Turn 2: open, close, open again, close, answer. Then a new segment.
    tokens = jnp.array([[HDR, OPEN, 9, CLOSE, 9, 9, EOT, HDR, OPEN, CLOSE, OPEN, 9, CLOSE, 9, 9, 9]])
    segments = jnp.array([[0] * 14 + [1] * 2])
    weights = jnp.ones_like(tokens).at[0, 4].set(0)
    expected = np.zeros(16, dtype=bool)
    expected[[3, 5]] = True  # the close predicts the answer; untrained position 4 is out; eot ends the turn
    expected[[9, 12, 13]] = True  # a close then a reopen still penalizes a second close at 9
    # Position 11 predicts a close (its true next token), so it is out; segment 1 starts fresh at 14.
    np.testing.assert_array_equal(np.asarray(post_close_positions(tokens, segments, weights, CONFIG))[0], expected)


def test_unlikelihood_adds_weighted_log_one_minus_close_probability():
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 16), 6, 16, dtype=jnp.int32)
    tokens = tokens.at[:, 5].set(CLOSE)
    weights = jnp.ones(tokens.shape, jnp.float32)
    segments = jnp.zeros(tokens.shape, jnp.int32)
    mask = AttentionMask.causal().with_segment_ids(segments)
    positions = post_close_positions(tokens, segments, weights, CONFIG)
    with jax.set_mesh(compact_grug_mesh(replica_axis_size=1)):
        model = Transformer.init(MODEL, key=jax.random.PRNGKey(0))
        plain, _ = model.next_token_loss(tokens, weights, mask=mask, reduction="sum", return_router_metrics=True)
        penalized, metrics = model.next_token_loss(
            tokens,
            weights,
            mask=mask,
            reduction="sum",
            return_router_metrics=True,
            unlikelihood=TokenUnlikelihood(CLOSE, CONFIG.weight, positions),
        )
        p_close = jax.nn.softmax(model.logits(tokens, mask=mask).astype(jnp.float32), axis=-1)[..., CLOSE]
    expected = CONFIG.weight * jnp.sum(jnp.where(positions, -jnp.log1p(-p_close), 0.0))
    assert int(positions.sum()) == 2 * 11  # the close at 5 through the last position
    np.testing.assert_allclose(penalized - plain, expected, rtol=1e-4)
    np.testing.assert_allclose(metrics["train/unlikelihood_loss"], expected, rtol=1e-4)


def test_microbatched_step_matches_full_batch_step_with_unlikelihood():
    # Device count is process-global, so the two-device mesh runs in a fresh CPU process.
    script = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp
        import jmp
        import numpy as np
        import optax
        from levanter.grug.sharding import compact_grug_mesh
        from levanter.data.text.examples import GrugLmExample
        from experiments.grug_sft.special_token_train import CloseUnlikelihoodConfig, GrugTrainState, _make_train_step
        from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer

        config = GrugModelConfig(
            vocab_size=16, hidden_dim=8, intermediate_dim=8,
            shared_expert_intermediate_dim=8, num_experts=2, num_experts_per_token=1,
            num_layers=2, num_heads=2, num_kv_heads=2, max_seq_len=8, sliding_window=4,
            attention_implementation="reference", moe_implementation="scatter",
            disable_pko=True, disable_long_rope=True, use_array_stacked_blocks=True,
        )
        unlikelihood = CloseUnlikelihoodConfig(3, 2, (1, 4, 5), weight=5.0)
        tokens = jax.random.randint(jax.random.PRNGKey(1), (4, 8), 6, 16, dtype=jnp.int32)
        tokens = tokens.at[:, 2].set(3).at[1, 5].set(5)
        segment_ids = jnp.array([[0] * 8, [0] * 4 + [1] * 4, [0] * 6 + [-1] * 2, [0] * 3 + [1] * 5], jnp.int32)
        weights = jax.random.bernoulli(jax.random.PRNGKey(2), 0.7, (4, 8)).astype(jnp.float32)
        batch = jax.vmap(GrugLmExample.causal)(tokens, segment_ids=segment_ids, loss_weight=weights)
        optimizer = optax.adam(1e-2)
        mp = jmp.get_policy("float32")
        results = []
        with jax.set_mesh(compact_grug_mesh(replica_axis_size=1)):
            for microbatches in (1, 2):
                params = Transformer.init(config, key=jax.random.PRNGKey(0))
                state = GrugTrainState(
                    step=jnp.array(0, jnp.int32), params=params, opt_state=optimizer.init(params),
                    ema_params=None, pending_qb_betas=jnp.zeros((config.num_layers, config.num_experts)),
                )
                step = _make_train_step(
                    optimizer, mp, z_loss_weight=1e-4, ema_beta=None, microbatches=microbatches,
                    close_unlikelihood=unlikelihood,
                )
                state, metrics, _ = step(state, batch)
                results.append((
                    metrics["train/loss"], metrics["train/unlikelihood_loss"], metrics["train/unlikelihood_mean_p"],
                    state.params,
                ))
                assert int(metrics["train/unlikelihood_positions"]) > 0
        assert float(results[0][1]) > 0
        for expected, actual in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1]), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        """
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_NUM_CPU_DEVICES": "2"}
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
