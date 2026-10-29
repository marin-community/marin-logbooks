# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

from experiments.post_training.identity_path_b.exposure import GLOBAL_BATCH, SEQUENCE_LENGTH, pack_arm, target_counts


def test_packed_dose_separates_actual_reasoning_and_final_tokens_and_source_occurrences(tmp_path):
    identity = []
    for source in range(16):
        for thinking in (False, True):
            prefix = [100] * 2048
            final = [42, 43, 44, 128009]
            trace = [128002, *range(51, 59), 128003] if thinking else []
            identity.append(
                {
                    "id": f"identity-{source}-{'on' if thinking else 'off'}",
                    "source_id": f"source-{source}",
                    "family": "name",
                    "enable_thinking": thinking,
                    "normalized_answer": "canonical name",
                    "input_ids": prefix + trace + final,
                    "assistant_masks": [0] * len(prefix) + [1] * (len(trace) + len(final)),
                }
            )
    replay = [
        {
            "id": f"replay-{index}",
            "source_id": f"replay-source-{index}",
            "family": "math",
            "enable_thinking": True,
            "input_ids": [100] * 2048 + [128002, *range(60, 80), 128003, 81, 82, 128009],
            "assistant_masks": [0] * 2048 + [1] * 25,
        }
        for index in range(32)
    ]
    output = tmp_path / "arm"
    report = pack_arm(identity, replay, identity_share=0.25, final_identity_minimum=30, output=output)
    by_id = {row["id"]: row for row in [*identity, *replay]}
    rows = pq.read_table(output / "packed.parquet").to_pylist()
    metadata = [json.loads(line) for line in (output / "sequence-rows.jsonl").read_text().splitlines()]
    final, reasoning, raw, source_counts = Counter(), Counter(), Counter(), Counter()
    for packed, binding in zip(rows, metadata, strict=True):
        assert len(packed["input_ids"]) == SEQUENCE_LENGTH
        position = 0
        for segment, row_id in enumerate(binding["ids"]):
            original = by_id[row_id]
            length = len(original["input_ids"])
            assert packed["input_ids"][position : position + length] == original["input_ids"]
            assert packed["segment_ids"][position : position + length] == [segment] * length
            expected = [
                float(mask) * (0.1 if token in (128002, 128003) else 1.0)
                for token, mask in zip(original["input_ids"], original["assistant_masks"], strict=True)
            ]
            np.testing.assert_allclose(packed["target_weights"][position : position + length], expected, rtol=1e-7)
            source = "identity" if row_id.startswith("identity-") else "replay"
            dose = target_counts(original)
            final[source] += dose.final_text
            reasoning[source] += dose.reasoning_text
            raw[source] += dose.raw
            if source == "identity":
                source_counts[original["source_id"]] += 1
                # Independent expected census for these deliberately different lengths.
                assert dose.final_text == 3
                assert dose.reasoning_text == (8 if original["enable_thinking"] else 0)
            position += length
        assert packed["segment_ids"][position:] == [-1] * (SEQUENCE_LENGTH - position)
        assert not any(packed["target_weights"][position:])
    assert len(rows) == report["updates"] * GLOBAL_BATCH
    assert report["final_text_tokens"] == dict(final)
    assert report["reasoning_text_tokens"] == dict(reasoning)
    assert report["raw_supervised_tokens"] == dict(raw)
    assert final["identity"] >= 30
    assert report["identity_source_occurrences"] == dict(source_counts)
    assert report["maximum_identity_source_occurrences_across_modes"] == max(source_counts.values()) <= 3
    assert report["cumulative_batches"][-1]["identity_final_text_tokens"] == final["identity"]
    assert report["cumulative_batches"][-1]["identity_reasoning_text_tokens"] == reasoning["identity"]
    assert report["structural_target_tokens_by_source"]["identity/128003"] == reasoning["identity"] // 8
