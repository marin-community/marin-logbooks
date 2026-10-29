# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from experiments.post_training.identity_path_b.exposure import pack_arm
from experiments.post_training.identity_path_b.trace_schedule import (
    checked_replacement,
    compact_row,
    load_variant,
    pack_schedule,
)


def source_rows():
    identity = []
    for source in range(32):
        for thinking in (False, True):
            prefix = [100] * 4086
            target = ([128002, 128003] if thinking else []) + [42, 43, 44, 128009]
            identity.append(
                {
                    "id": f"identity-{source}-{'on' if thinking else 'off'}",
                    "source_id": f"source-{source}",
                    "family": "name",
                    "enable_thinking": thinking,
                    "normalized_answer": "canonical name",
                    "input_ids": prefix + target,
                    "assistant_masks": [0] * len(prefix) + [1] * len(target),
                    "target_start": len(prefix),
                    "messages": [
                        {"role": "user", "content": "Your name?"},
                        {
                            "role": "assistant",
                            "content": ("<|start_think|><|end_think|>" if thinking else "") + "canonical name",
                        },
                    ],
                }
            )
    replay = [
        {
            "id": f"replay-{i}",
            "source_id": f"replay-source-{i}",
            "family": "math",
            "enable_thinking": True,
            "input_ids": [100] * 4080 + [128002, 61, 62, 128003, 81, 128009],
            "assistant_masks": [0] * 4080 + [1] * 6,
        }
        for i in range(64)
    ]
    return identity, replay


def test_fixed_ledger_fallback_matches_base_and_traces_repack_without_new_occurrences(tmp_path):
    identity, replay = source_rows()
    base_path = tmp_path / "base"
    baseline = pack_arm(identity, replay, identity_share=0.25, final_identity_minimum=6, output=base_path)
    baseline["identity_original_sources"] = 32
    rows = {r["id"]: compact_row(r) for r in [*identity, *replay]}
    fallback = tmp_path / "fallback"
    fallback_report = pack_schedule(rows, base_path / "occurrence-ledger.jsonl", baseline, fallback)
    assert pq.read_table(fallback / "packed.parquet").equals(pq.read_table(base_path / "packed.parquet"))
    assert (fallback / "sequence-rows.jsonl").read_bytes() == (base_path / "sequence-rows.jsonl").read_bytes()
    assert fallback_report["step_count_change"] == 0
    assert fallback_report["final_text_tokens"] == baseline["final_text_tokens"]

    for row in identity:
        if not row["enable_thinking"]:
            continue
        replacement = copy.deepcopy(row)
        start = row["target_start"]
        replacement["input_ids"] = row["input_ids"][: start + 1] + [55] * 20 + row["input_ids"][start + 1 :]
        replacement["assistant_masks"] = row["assistant_masks"][:start] + [1] * (len(replacement["input_ids"]) - start)
        replacement["messages"][-1] = {
            "role": "assistant",
            "content": "canonical name",
            "reasoning_content": "rationale",
        }
        rows[row["id"]] = checked_replacement(row, replacement)
    traced = tmp_path / "traced"
    report = pack_schedule(rows, base_path / "occurrence-ledger.jsonl", baseline, traced)
    assert (traced / "occurrence-ledger.jsonl").read_bytes() == (base_path / "occurrence-ledger.jsonl").read_bytes()
    bindings = [json.loads(line) for line in (traced / "sequence-rows.jsonl").read_text().splitlines()]
    ledger = [json.loads(line) for line in (base_path / "occurrence-ledger.jsonl").read_text().splitlines()]
    assert [row_id for binding in bindings for row_id in binding["ids"]] == [r["row_id"] for r in ledger]
    encountered_on = sum(r["source"] == "identity" and r["enable_thinking"] for r in ledger)
    assert report["reasoning_text_tokens"]["identity"] == 20 * encountered_on
    assert (
        report["raw_supervised_tokens"]["identity"]
        == baseline["raw_supervised_tokens"]["identity"] + 20 * encountered_on
    )
    assert report["weighted_mass"]["identity"] == pytest.approx(
        baseline["weighted_mass"]["identity"] + 20 * encountered_on
    )
    assert report["final_text_tokens"] == baseline["final_text_tokens"]
    assert report["updates"] > baseline["updates"]
    assert report["identity_trace_encounters"]["on_accepted_trace"] == encountered_on
    assert report["cumulative_batches"][-1]["identity_reasoning_text_tokens"] == 20 * encountered_on
    packed = pq.read_table(traced / "packed.parquet").to_pylist()
    assert len(packed) == report["updates"] * 8
    assert bindings[-1]["ids"] == []  # Padding adds no training occurrence.
    assert packed[-1]["segment_ids"] == [-1] * 8192
    assert not any(packed[-1]["target_weights"])
    for sequence, binding in zip(packed, bindings, strict=True):
        position = 0
        for segment, row_id in enumerate(binding["ids"]):
            row = rows[row_id]
            n = len(row.input_ids)
            np.testing.assert_array_equal(sequence["input_ids"][position : position + n], row.input_ids)
            assert sequence["segment_ids"][position : position + n] == [segment] * n
            position += n


def test_trace_graft_cannot_change_final_tokens_even_with_same_lexical_dose():
    identity, _ = source_rows()
    base = next(r for r in identity if r["enable_thinking"])
    replacement = copy.deepcopy(base)
    start = base["target_start"]
    replacement["input_ids"] = base["input_ids"][: start + 1] + [55] * 20 + base["input_ids"][start + 1 :]
    replacement["assistant_masks"] = [0] * start + [1] * (len(replacement["input_ids"]) - start)
    replacement["messages"][-1]["content"] = "canonical name"
    replacement["input_ids"][-2] = 45
    with pytest.raises(ValueError, match="canonical final IDs"):
        checked_replacement(base, replacement)


def test_variant_keeps_off_and_fallback_line_bytes_and_changes_only_selected_on(tmp_path):
    identity, replay = source_rows()
    original_lines = [(json.dumps(row, separators=(", ", ": ")) + "\n").encode() for row in identity]
    identity_path, replay_path, drafts = (tmp_path / name for name in ("identity.jsonl", "replay.jsonl", "drafts.jsonl"))
    identity_path.write_bytes(b"".join(original_lines))
    replay_bytes = b"".join((json.dumps(row) + "\n").encode() for row in replay)
    replay_path.write_bytes(replay_bytes)
    base = next(r for r in identity if r["enable_thinking"])
    trace = copy.deepcopy(base)
    start = base["target_start"]
    trace["input_ids"] = base["input_ids"][: start + 1] + [55] * 20 + base["input_ids"][start + 1 :]
    trace["assistant_masks"] = [0] * start + [1] * (len(trace["input_ids"]) - start)
    trace["messages"][-1] = {"role": "assistant", "content": "canonical name", "reasoning_content": "rationale"}
    trace_line = (json.dumps(trace, sort_keys=True) + "\n").encode()
    drafts.write_bytes(trace_line)
    output = tmp_path / "variant.jsonl"
    rows, report = load_variant(
        identity_path,
        replay_path,
        [drafts],
        {base["source_id"]},
        output,
        identity_sha256=hashlib.sha256(identity_path.read_bytes()).hexdigest(),
        replay_sha256=hashlib.sha256(replay_bytes).hexdigest(),
    )
    assert output.read_bytes().splitlines(keepends=True) == [
        trace_line if row["id"] == base["id"] else line for row, line in zip(identity, original_lines, strict=True)
    ]
    assert len(rows) == len(identity) + len(replay)
    assert report["identity_row_dispositions"] == {
        "off_exact_unchanged": 32,
        "on_exact_empty_fallback": 31,
        "on_trace_replacements": 1,
    }
    assert rows[base["id"]].dose.reasoning_text == 20
