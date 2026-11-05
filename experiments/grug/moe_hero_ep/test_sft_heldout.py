# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe_hero_ep.format_eval import EOT_ID
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.sft import SftConfig, initialize_sft, merge_sft, sft_logits
from experiments.grug.moe_hero_ep.sft_heldout import (
    HeldoutCase,
    HeldoutPanel,
    compare_export_logits,
    heldout_batches,
    merged_probe_logits,
    select_checkpoint,
    teacher_forced_metrics,
)


def test_heldout_batches_preserve_causal_assistant_targets_and_exclude_filler():
    panel = HeldoutPanel(
        source_manifest_digest="manifest",
        heldout_sha256="heldout",
        train_sha256="train",
        tokenizer="tokenizer",
        tokenizer_revision="revision",
        template_sha256="template",
        cases=(
            HeldoutCase(id="a", token_ids=(1, 2, EOT_ID), next_token_weights=(0, 1, 0)),
            HeldoutCase(id="b", token_ids=(3, 4, 5, EOT_ID), next_token_weights=(1, 0, 1, 0)),
        ),
    )
    cases, tokens, weights, segments, positions = next(heldout_batches(panel, 4, 8))
    labels = np.pad(tokens[:, 1:], ((0, 0), (0, 1)))
    np.testing.assert_array_equal(labels[weights > 0], [EOT_ID, 4, EOT_ID])
    np.testing.assert_array_equal(labels[np.arange(len(cases)), positions[: len(cases)]], [EOT_ID, EOT_ID])
    np.testing.assert_array_equal(weights[2:], np.zeros((2, 8)))
    np.testing.assert_array_equal(segments[2:], np.full((2, 8), -1))
    assert int(weights.sum()) == 3


def test_checkpoint_selection_preserves_eot_and_prefers_earlier_nll_tie():
    measurements = [
        {"step": 64, "eot_top1": 14, "assistant_nll": 2.0},
        {"step": 96, "eot_top1": 14, "assistant_nll": 1.9},
        {"step": 128, "eot_top1": 13, "assistant_nll": 1.8},
    ]
    assert select_checkpoint(measurements) == 96
    measurements[2]["eot_top1"] = 15
    measurements[2]["assistant_nll"] = 1.9
    assert select_checkpoint(measurements) == 96
    measurements[1]["assistant_nll"] = 2.1
    measurements[2]["assistant_nll"] = 2.2
    assert select_checkpoint(measurements) == 64


def test_merge_diagnostics_distinguish_logit_offsets_from_probability_changes():
    compact = np.array([[1, 2, 0], [0, 1, 0]], dtype=np.float32)
    merged = np.array([[4, 5, 3], [2, 1, 0]], dtype=np.float32)
    observed = compare_export_logits(compact, merged)
    assert observed["finite"]
    assert observed["max_absolute_logit_delta"] == 3
    assert observed["greedy_disagreements"] == 1
    np.testing.assert_allclose(observed["mean_absolute_logit_delta"], 11 / 6)
    np.testing.assert_allclose(observed["max_absolute_probability_delta"], 0.453299398, atol=1e-9)
    shifted_only = compare_export_logits(compact[:1], merged[:1])
    assert shifted_only["max_absolute_logprob_delta"] == 0
    assert not compare_export_logits(compact, np.full_like(compact, np.nan))["finite"]


def test_heldout_teacher_forcing_matches_dense_logits_with_masked_padding():
    config = GrugModelConfig(
        vocab_size=128256,
        hidden_dim=8,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_shared_experts=1,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        local_kv_heads=1,
        global_kv_heads=1,
        head_dim=4,
        max_seq_len=4,
        sliding_window=4,
        global_every=2,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
        capacity_factor=4.0,
    )
    sft = SftConfig(2, 2.0, (2, EOT_ID), 1e-3, 1e-3, 128, 2, 1, token_block_size=4)
    with jax.set_mesh(compact_grug_mesh(replica_axis_size=1)):
        base = Transformer.init(config, key=jax.random.key(0))
        frozen, state = initialize_sft(base, sft)
        params = eqx.tree_at(
            lambda value: value.output_delta,
            state.params,
            jnp.full(state.params.output_delta.shape, 0.1, jnp.float32),
        )
        tokens = jnp.array([[1, 2, EOT_ID, 0], [3, 4, 2, EOT_ID]], jnp.int32)
        weights = jnp.array([[0, 1, 0, 0], [1, 1, 1, 0]], jnp.float32)
        segments = jnp.array([[0, 0, 0, -1], [0, 0, 0, 0]], jnp.int32)
        positions = jnp.array([1, 2], jnp.int32)
        actual_loss, _, actual_eot, actual_top1 = teacher_forced_metrics(
            frozen, params, tokens, weights, segments, positions, sft
        )
        dense = sft_logits(frozen, params, tokens, sft, AttentionMask.causal().with_segment_ids(segments))
        logits = np.asarray(dense, np.float64)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        logprobs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        labels = np.pad(np.asarray(tokens)[:, 1:], ((0, 0), (0, 1)))
        target_logprobs = np.take_along_axis(logprobs, labels[..., None], axis=-1)[..., 0]
        expected_loss = -np.sum(target_logprobs * np.asarray(weights)) / np.asarray(weights).sum()
        expected_eot = -logprobs[np.arange(2), np.asarray(positions), EOT_ID]
        expected_top1 = np.argmax(logprobs[np.arange(2), np.asarray(positions)], axis=-1)
        np.testing.assert_allclose(actual_loss, expected_loss, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(actual_eot, expected_eot, atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(actual_top1, expected_top1)
        merged = merge_sft(frozen, params, sft)
        ordinary = merged_probe_logits(merged, tokens, segments, positions)
        hidden, _ = merged(tokens, AttentionMask.causal().with_segment_ids(segments))
        selected_hidden = np.asarray(hidden, np.float64)[np.arange(2), np.asarray(positions)]
        head = np.asarray(merged.output_proj, np.float64)
        expected_ordinary = selected_hidden @ head
        epsilon = np.finfo(np.float32).eps
        gamma = config.hidden_dim * epsilon / (1 - config.hidden_dim * epsilon)
        bound = gamma * (np.abs(selected_hidden) @ np.abs(head))
        assert np.all(np.abs(np.asarray(ordinary) - expected_ordinary) <= bound)
