# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh
from levanter.store.cache import write_levanter_cache
from levanter.testing.devices import use_test_devices
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from experiments.grug.moe_hero_ep import model
from experiments.grug.moe_hero_ep.replay_loss_eval import (
    REPLAY_LOSS_METRIC_PREFIX,
    ReplayDocumentLoss,
    ReplayLossEvalConfig,
    replay_loss_metrics,
    score_replay_documents,
)

SEQ_LEN = 16
CONFIG = model.GrugModelConfig(
    vocab_size=64,
    hidden_dim=32,
    intermediate_dim=16,
    shared_expert_intermediate_dim=16,
    num_shared_experts=1,
    num_experts=4,
    num_experts_per_token=2,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    local_kv_heads=2,
    global_kv_heads=2,
    head_dim=8,
    max_seq_len=SEQ_LEN,
    sliding_window=4,
    global_every=2,
    initializer_std=0.1,
    qk_mult=1.3,
    capacity_factor=8.0,
    sconv=True,
    rope_fused=True,
    attention_implementation="reference",
    moe_implementation="fixed_all_to_all",
)
CATEGORIES = [None, "structured_data", None, "minified_code", None, None, "repetition", None]


def _write_fixture(directory):
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, **{str(i): i for i in range(1, 64)}}, unk_token="<unk>"))
    tokenizer.save(f"{directory}/tokenizer.json")
    rows = [np.random.default_rng(i).integers(2, 64, size=SEQ_LEN).astype(np.int32) for i in range(len(CATEGORIES))]
    write_levanter_cache(iter({"input_ids": row} for row in rows), f"{directory}/data/validation", metadata={})
    manifest = [
        {"cell": f"c{i:02d}q0-validation", "shard": "part-00207", "index": 10 * i, "category": category}
        for i, category in enumerate(CATEGORIES)
    ]
    (directory / "data" / "manifest.jsonl").write_text("".join(json.dumps(record) + "\n" for record in manifest))
    return rows, manifest


def _reference_loss(weights, tokens):
    """Score one document alone in its batch, every position but the last predicting the next token."""
    ids = np.zeros((4, SEQ_LEN), dtype=np.int32)
    ids[0] = tokens
    weight = np.zeros((4, SEQ_LEN), dtype=np.float32)
    weight[0, : SEQ_LEN - 1] = 1.0
    expert_mlp = weights.stacked_blocks.stacked.mlp.expert_mlp
    local = eqx.tree_at(
        lambda m: m.stacked_blocks.stacked.mlp.expert_mlp,
        weights,
        dataclasses.replace(expert_mlp, implementation="scatter", expert_chunks=1),
    )
    losses = local.next_token_loss(
        jnp.asarray(ids), jnp.asarray(weight), mask=model.AttentionMask.causal(), reduction="none", logsumexp_weight=None
    )
    return float(np.sum(np.asarray(losses)[0] * weight[0]))


@pytest.mark.timeout(300)
def test_each_row_is_one_whole_document_bound_to_its_manifest_line(tmp_path):
    rows, manifest = _write_fixture(tmp_path)
    with use_test_devices(8):
        mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, context_axis_size=2)
        with set_mesh(mesh):
            params = model.Transformer.init(CONFIG, key=jax.random.key(3))
            scored = score_replay_documents(
                params,
                ReplayLossEvalConfig(
                    data_path=str(tmp_path / "data"),
                    output_path=str(tmp_path),
                    tokenizer=str(tmp_path),
                    batch_size=4,
                    seq_len=SEQ_LEN,
                ),
                mesh=mesh,
                mp=jmp.get_policy("float32"),
                implementation="scatter",
            )
            assert [(s.cell, s.index, s.category) for s in scored] == [
                (m["cell"], m["index"], m["category"]) for m in manifest
            ]
            for result, tokens in zip(scored, rows, strict=True):
                assert result.scored_tokens == SEQ_LEN - 1
                np.testing.assert_allclose(result.loss_sum, _reference_loss(params, tokens), rtol=2e-4)


def test_metrics_split_documents_by_verdict():
    def row(category, loss_sum, tokens):
        return ReplayDocumentLoss("c", "s", 0, category, tokens, loss_sum, tokens * 4)

    metrics = replay_loss_metrics([row(None, 10.0, 10), row(None, 30.0, 10), row("structured_data", 5.0, 20)])
    assert metrics[f"{REPLAY_LOSS_METRIC_PREFIX}/kept/loss"] == pytest.approx(2.0)
    assert metrics[f"{REPLAY_LOSS_METRIC_PREFIX}/flagged/loss"] == pytest.approx(0.25)
    assert metrics[f"{REPLAY_LOSS_METRIC_PREFIX}/all/loss"] == pytest.approx(45.0 / 40)
    assert metrics[f"{REPLAY_LOSS_METRIC_PREFIX}/flagged/documents"] == 1.0
