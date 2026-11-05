# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import draccus
import jax
import jax.numpy as jnp
import numpy as np
import optax
from levanter.checkpoint import save_checkpoint
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.grug.moe_hero_ep.sft import (
    DONOR_COLUMNS,
    BaseReference,
    SftConfig,
    _local_cross_entropy,
    compact_optimizer_report,
    initialize_from_checkpoint,
    initialize_sft,
    make_sft_step,
    restore_sft_checkpoint,
    save_sft_checkpoint,
    sft_logits,
)
from experiments.grug.moe_hero_ep.weights import cast_weights


def model_config(vocab_size: int) -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=vocab_size,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=65536,
        sliding_window=4,
        global_every=2,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
        capacity_factor=4.0,
    )


def sft_config(token_ids: tuple[int, ...]) -> SftConfig:
    return SftConfig(2, 2.0, token_ids, 1e-3, 1e-3, 128, 2, 1, 32, 4)


def test_tiled_vocab_loss_and_gradients_match_dense_reference():
    hidden = jax.random.normal(jax.random.key(1), (1, 7, 8))
    head = jax.random.normal(jax.random.key(2), (8, 37))
    delta = jax.random.normal(jax.random.key(3), (8, 3)) * 0.1
    labels = jnp.array([[3, 5, 17, 25, 1, 36, 0]])
    weights = jnp.array([[0, 1, 1, 1, 0, 1, 0]], jnp.float32)
    selected = jnp.array([3, 17, 36])

    def tiled(x, d):
        total, count = _local_cross_entropy(
            x, head, labels, weights, d, selected, vocab_block_size=16, token_block_size=4
        )
        return total / count

    def reference(x, d):
        logits = x @ head + (x @ d) @ jax.nn.one_hot(selected, 37)
        return jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits, labels) * weights) / jnp.sum(weights)

    actual = jax.jit(jax.value_and_grad(tiled, argnums=(0, 1)))(hidden, delta)
    expected = jax.value_and_grad(reference, argnums=(0, 1))(hidden, delta)
    for value, reference_value in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(value, reference_value, atol=2e-6, rtol=2e-6)


def test_short_packed_updates_resume_with_the_same_128_update_schedule(tmp_path: Path):
    config = model_config(64)
    sft = sft_config((2, 3, 4, 5, 6))
    reference = BaseReference(
        "source",
        "metadata",
        390251,
        "model",
        "config",
        "manifest",
        "tokenizer",
        "revision",
        "evidence",
        "source-config",
        "source-revision",
    )
    with jax.set_mesh(compact_grug_mesh()):
        base = Transformer.init(config, key=jax.random.key(0))
        frozen, initial = initialize_sft(base, sft)
        tokens = jnp.array([[2, 4, 1, 4, 2, 3, 5, 7]], jnp.int32)
        weights = jnp.ones_like(tokens, dtype=jnp.float32).at[:, -1].set(0)
        valid = jnp.ones_like(tokens, dtype=bool)
        np.testing.assert_array_equal(
            sft_logits(frozen, initial.params, tokens, sft), cast_weights(base, jnp.bfloat16).logits(tokens)
        )
        update = make_sft_step(sft)
        state = initial
        for _ in range(3):
            state, metrics = update(state, frozen, tokens, weights, valid)
        checkpoint = str(tmp_path / "adapters")
        save_sft_checkpoint(checkpoint, state, reference, sft)
        restored = restore_sft_checkpoint(checkpoint, initial, reference, sft, jax.sharding.get_mesh())
        expected, _ = update(state, frozen, tokens, weights, valid)
        actual, _ = update(restored, frozen, tokens, weights, valid)
        for value, reference_value in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(value, reference_value)
        assert int(actual.step) == 4 and int(actual.processed_tokens) == 32
        assert compact_optimizer_report(actual)["compact_optimizer_only"]
        assert np.isfinite(float(metrics["loss"]))
        assert np.linalg.norm(np.asarray(actual.params.output_delta)) > 0
        assert frozen.config.max_seq_len == 65536


def test_direct_source_restore_initializes_only_five_donor_output_columns(tmp_path: Path):
    config = model_config(128256)
    sft = sft_config(tuple(target for target, _ in DONOR_COLUMNS))
    source = tmp_path / "source"
    with jax.set_mesh(compact_grug_mesh()):
        base = Transformer.init(config, key=jax.random.key(0))
        pending = jnp.linspace(-0.013, 0.017, config.num_layers * config.num_experts).reshape(
            config.num_layers, config.num_experts
        )
        save_checkpoint({"params": base, "pending_qb_betas": pending}, 390251, str(source), is_temporary=False)
        config_path = tmp_path / "model.json"
        encoded_config = draccus.encode(config)
        config_path.write_text(json.dumps(encoded_config))
        reference = BaseReference(
            str(source),
            digest(json.loads((source / "metadata.json").read_text())),
            390251,
            str(config_path),
            digest(encoded_config),
            digest(json.loads((source / "manifest.json").read_text())),
            "tokenizer",
            "revision",
            "verified source tokenizer",
            "source-config-digest",
            "source-revision",
        )
        frozen, _ = initialize_from_checkpoint(reference, config, jax.sharding.get_mesh(), sft)
        expected = np.asarray(base.output_proj.astype(jnp.bfloat16)).copy()
        for target, donor in DONOR_COLUMNS:
            expected[:, target] = np.asarray(base.output_proj.astype(jnp.bfloat16))[:, donor]
        np.testing.assert_array_equal(frozen.output_proj, expected)
        np.testing.assert_array_equal(frozen.token_embed, base.token_embed.astype(jnp.bfloat16))
        np.testing.assert_array_equal(
            frozen.stacked_blocks.stacked.mlp.router_bias, -pending + pending.mean(axis=-1, keepdims=True)
        )
