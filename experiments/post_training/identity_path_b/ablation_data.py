# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finite fixed-horizon ablation datasets derived without sampling or target edits."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.post_training.identity_path_b.exposure import GLOBAL_BATCH, SEQUENCE_LENGTH
from experiments.post_training.identity_path_b.trace_schedule import ScheduledRow

UPDATES = 36
SEQUENCES = UPDATES * GLOBAL_BATCH
SCHEMA = pa.schema(
    [
        ("input_ids", pa.list_(pa.int32())),
        ("target_weights", pa.list_(pa.float32())),
        ("segment_ids", pa.list_(pa.int32())),
    ]
)


def ordered_prefix(parent: Path) -> list[dict]:
    sequences = [json.loads(line) for line in (parent / "sequence-rows.jsonl").read_bytes().splitlines()][:SEQUENCES]
    identifiers = [identifier for row in sequences for identifier in row["ids"]]
    ledger = [json.loads(line) for line in (parent / "occurrence-ledger.jsonl").read_bytes().splitlines()]
    prefix = ledger[: len(identifiers)]
    if identifiers != [row["row_id"] for row in prefix]:
        raise ValueError("The 36-update packed prefix does not join its original ordered ledger")
    return prefix


def off_identity_occurrences(schedule: list[dict], rows: dict[str, ScheduledRow]) -> list[dict]:
    pairs = {row.source_id: row for row in rows.values() if not row.enable_thinking}
    result = []
    for occurrence in schedule:
        row = occurrence
        if row["source"] == "identity" and row["enable_thinking"]:
            off = pairs[row["source_id"]]
            if off.dose.final_text != row["canonical_final_text_tokens"]:
                raise ValueError("Paired OFF changes the canonical final dose")
            row = dict(row, row_id=off.row_id, enable_thinking=False)
        result.append(row)
    return result


def pack_prefix(
    rows: dict[str, ScheduledRow], schedule: list[dict], output: Path, *, truncate: bool, updates: int = UPDATES
) -> dict:
    """Greedy whole-conversation packing; no crop, repeat or extra source selection.

    A0/A1/A3 consume their exact finite occurrence list; A2 consumes the longest
    replay-only prefix fitting the requested horizon. Padding completes the final batch.
    """
    sequences_required = updates * GLOBAL_BATCH
    output.mkdir(parents=True, exist_ok=False)
    selected, sequences, metadata, cumulative = [], [], [], []
    tokens, masks, segments, identifiers = [], [], [], []
    raw, final, reasoning, family, modes, structural = (Counter() for _ in range(6))
    repeats, rendered, families = Counter(), Counter(), Counter()

    def summary():
        return {
            "completed_updates": len(sequences) // GLOBAL_BATCH,
            "identity_final_text_tokens": final["identity"],
            "identity_supervised_tokens": raw["identity"],
            "all_supervised_tokens": raw.total(),
            "raw_supervised_tokens": dict(raw),
            "final_text_tokens": dict(final),
            "reasoning_text_tokens": dict(reasoning),
            "mode_final_text_tokens": dict(modes),
            "family_final_text_tokens": dict(family),
            "structural_target_tokens": dict(structural),
        }

    def flush():
        nonlocal tokens, masks, segments, identifiers
        count = len(tokens)
        sequences.append(
            {
                "input_ids": tokens + [128001] * (SEQUENCE_LENGTH - count),
                "target_weights": masks + [0.0] * (SEQUENCE_LENGTH - count),
                "segment_ids": segments + [-1] * (SEQUENCE_LENGTH - count),
            }
        )
        metadata.append({"sequence": len(sequences) - 1, "valid_positions": count, "ids": identifiers})
        tokens, masks, segments, identifiers = [], [], [], []
        if len(sequences) % GLOBAL_BATCH == 0:
            cumulative.append(summary())

    for occurrence in schedule:
        row = rows[occurrence["row_id"]]
        if row.source_id != occurrence["source_id"] or row.enable_thinking != occurrence["enable_thinking"]:
            raise ValueError("Occurrence/source/mode mismatch")
        size = len(row.input_ids)
        if size > SEQUENCE_LENGTH:
            raise ValueError("Ablations cannot crop long source rows")
        if tokens and len(tokens) + size > SEQUENCE_LENGTH:
            flush()
        if len(sequences) == sequences_required:
            if truncate:
                break
            raise ValueError(f"The exact occurrence schedule no longer fits{updates} updates")
        index = len(identifiers)
        tokens.extend(row.input_ids.tolist())
        masks.extend(row.assistant_masks.astype(float).tolist())
        segments.extend([index] * size)
        identifiers.append(row.row_id)
        source = occurrence["source"]
        dose = row.dose
        raw[source] += dose.raw
        final[source] += dose.final_text
        reasoning[source] += dose.reasoning_text
        family[f"{source}/{row.family}"] += dose.final_text
        families[f"{source}/{row.family}"] += 1
        modes[f"{source}/{'on' if row.enable_thinking else 'off'}"] += dose.final_text
        structural.update({f"{source}/{token}": n for token, n in dose.structural.items()})
        if source == "identity":
            repeats[row.source_id] += 1
            rendered[row.row_id] += 1
        selected.append(dict(occurrence, occurrence=len(selected)))
    if tokens:
        flush()
    if truncate and len(sequences) < sequences_required:
        raise ValueError(f"Source stream cannot fill the fixed{updates}-update prefix")
    while len(sequences) < sequences_required:
        flush()
    if len(sequences) != sequences_required:
        raise ValueError(f"Expected exactly{updates} batches of8")
    pq.write_table(pa.Table.from_pylist(sequences, schema=SCHEMA), output / "packed.parquet", compression="zstd")
    for name, records in [("occurrence-ledger.jsonl", selected), ("sequence-rows.jsonl", metadata)]:
        (output / name).write_text("".join(json.dumps(row) + "\n" for row in records))
    report = {
        "experiment": "PathB-BFCL-ablation",
        "updates": updates,
        "checkpoint_updates": [updates],
        "batch_size": GLOBAL_BATCH,
        "sequence_length": SEQUENCE_LENGTH,
        "special_token_loss_weight": 1.0,
        "raw_supervised_tokens": dict(raw),
        "weighted_mass": dict(raw),
        "final_text_tokens": dict(final),
        "reasoning_text_tokens": dict(reasoning),
        "family_final_text_tokens": dict(family),
        "family_row_occurrences": dict(families),
        "mode_final_text_tokens": dict(modes),
        "structural_target_tokens": dict(structural),
        "cumulative_batches": cumulative,
        "row_occurrences": len(selected),
        "padding_only_sequences": sum(not r["ids"] for r in metadata),
        "maximum_identity_source_occurrences_across_modes": max(repeats.values(), default=0),
        "maximum_identity_rendered_row_occurrences": max(rendered.values(), default=0),
        "realized_identity_supervised_share": raw["identity"] / raw.total(),
        "ordered_occurrence_ledger_sha256": (
            hashlib.sha256((output / "occurrence-ledger.jsonl").read_bytes()).hexdigest()
        ),
    }
    (output / "exposure.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
