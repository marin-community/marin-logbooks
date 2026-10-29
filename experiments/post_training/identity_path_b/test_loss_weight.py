# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weight-only intervention: source classification and frozen rows remain intact."""

import numpy as np
import pyarrow.parquet as pq

from experiments.post_training.identity_path_b.data import END_IDS, LOW_WEIGHT_IDS
from experiments.post_training.identity_path_b.exposure import pack_arm
from experiments.post_training.identity_path_b.recipe import FROZEN_TOKEN_IDS
from experiments.post_training.identity_path_b.test_trace_schedule import source_rows
from experiments.post_training.identity_path_b.trace_schedule import compact_row, pack_schedule


def test_weight_one_changes_only_supervised_structural_weights_in_both_sources(tmp_path):
    identity, replay = source_rows()
    # Actual tool-call/end-tool controls in replay, in addition to native think delimiters.
    for row in replay:
        row["input_ids"][-5:-3] = [128005, 128011]
    original = tmp_path / "old"
    baseline = pack_arm(identity, replay, identity_share=0.25, final_identity_minimum=6, output=original)
    baseline["identity_original_sources"] = 32
    rows = {r["id"]: compact_row(r) for r in [*identity, *replay]}
    new = tmp_path / "new"
    report = pack_schedule(rows, original / "occurrence-ledger.jsonl", baseline, new, special_token_loss_weight=1.0)
    assert report["weighted_mass"] == report["raw_supervised_tokens"]
    assert report["special_token_loss_weight"] == 1.0
    for key in ("updates", "final_text_tokens", "reasoning_text_tokens", "raw_supervised_tokens", "checkpoint_updates"):
        assert report[key] == baseline[key]
    for filename in ("occurrence-ledger.jsonl", "sequence-rows.jsonl"):
        assert (new / filename).read_bytes() == (original / filename).read_bytes()
    changed = set()
    for before, after in zip(
        pq.read_table(original / "packed.parquet").to_pylist(),
        pq.read_table(new / "packed.parquet").to_pylist(),
        strict=True,
    ):
        assert before["input_ids"] == after["input_ids"]
        assert before["segment_ids"] == after["segment_ids"]
        old = np.asarray(before["target_weights"])
        expected = (old > 0).astype(np.float32)
        np.testing.assert_array_equal(after["target_weights"], expected)
        changed.update(token for token, a, b in zip(after["input_ids"], old, expected, strict=True) if a != b)
    assert {128002, 128003, 128005, 128011} <= changed
    assert changed <= LOW_WEIGHT_IDS
    assert len(FROZEN_TOKEN_IDS) == 256
    assert set(FROZEN_TOKEN_IDS) == LOW_WEIGHT_IDS | END_IDS
