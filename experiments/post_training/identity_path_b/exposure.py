# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic whole-conversation packing and actual per-arm exposure accounting."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.post_training.identity_path_b.data import END_IDS, LOW_WEIGHT_IDS

SEQUENCE_LENGTH = 8192
GLOBAL_BATCH = 8


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class TargetDose:
    raw: int
    final_text: int
    reasoning_text: int
    weighted: float
    structural: Counter[str]


def target_counts(row: dict) -> TargetDose:
    """Count actual masked target IDs; reasoning never contributes to final-answer dose."""
    target = [t for t, m in zip(row["input_ids"], row["assistant_masks"], strict=True) if m]
    reasoning, final = [], target
    if target and target[0] == 128002:
        boundary = target.index(128003)
        reasoning, final = target[1:boundary], target[boundary + 1 :]
    return TargetDose(
        raw=len(target),
        final_text=sum(t not in LOW_WEIGHT_IDS | END_IDS for t in final),
        reasoning_text=sum(t not in LOW_WEIGHT_IDS | END_IDS for t in reasoning),
        weighted=sum(0.1 if t in LOW_WEIGHT_IDS else 1.0 for t in target),
        structural=Counter(str(t) for t in target if t in LOW_WEIGHT_IDS | END_IDS),
    )


@dataclass
class RowStream:
    rows: list[dict]
    cycle: int = 0
    offset: int = 0

    def next_row(self) -> dict:
        if self.offset == 0:
            self.rows = sorted(self.rows, key=lambda r: digest(f"310:{self.cycle}:{r['id']}"))
        row = self.rows[self.offset]
        self.offset += 1
        if self.offset == len(self.rows):
            self.offset = 0
            self.cycle += 1
        return row


class IdentityStream:
    """One epoch visits each original source once; modes alternate across epochs."""

    def __init__(self, rows: list[dict]):
        self.variants = defaultdict(dict)
        for row in rows:
            if row["enable_thinking"] in self.variants[row["source_id"]]:
                raise ValueError("Duplicate identity source/mode variant")
            self.variants[row["source_id"]][row["enable_thinking"]] = row
        if any(set(variants) != {False, True} for variants in self.variants.values()):
            raise ValueError("Every identity source needs exactly one off/on pair")
        ordered = sorted(self.variants, key=lambda source: digest(f"310:mode:{source}"))
        self.first_mode = {source: i >= len(ordered) // 2 for i, source in enumerate(ordered)}
        self.cycle, self.offset = 0, 0
        self.order = []

    def next_row(self) -> dict:
        if self.offset == 0:
            self.order = sorted(self.variants, key=lambda source: digest(f"310:{self.cycle}:{source}"))
        source = self.order[self.offset]
        thinking = self.first_mode[source] ^ bool(self.cycle % 2)
        row = self.variants[source][thinking]
        self.offset += 1
        if self.offset == len(self.order):
            self.offset = 0
            self.cycle += 1
        return row


def checkpoint_cadence(updates: int, epoch_steps: list[int]) -> list[int]:
    """Use completed-update labels; coalesce coincident quarter/epoch boundaries."""
    return sorted(set([math.ceil(updates * q / 4) for q in (1, 2, 3, 4)] + epoch_steps))


def pack_arm(
    identity: list[dict], replay: list[dict], *, identity_share: float, final_identity_minimum: int, output: Path
) -> dict:
    """Pack a finite arm, stopping only on an audited complete global batch.

    Share uses raw supervised tokens. Minimum dose excludes all structural/end
    tokens. Weighted mass and per-family/mode/row exposure are reported separately.
    """
    output.mkdir(parents=True, exist_ok=False)
    streams = {"identity": IdentityStream(identity), "replay": RowStream(replay)}
    total, final, reasoning, occurrences = Counter(), Counter(), Counter(), Counter()
    weighted, family_weighted = defaultdict(float), defaultdict(float)
    family, family_raw, family_reasoning = Counter(), Counter(), Counter()
    family_rows, mode, structural, answer_rows, answer_final = Counter(), Counter(), Counter(), Counter(), Counter()
    epoch_steps, batch_summaries, packed = [], [], []
    tokens, masks, segments, row_ids = [], [], [], []
    metadata = []
    source_pool = {r["id"]: target_counts(r) for r in [*identity, *replay]}
    if len(source_pool) != len(identity) + len(replay):
        raise ValueError("Identity/replay rendered IDs must be unique")
    unique_identity_final = sum(source_pool[row["id"]].final_text for row in identity if not row["enable_thinking"])
    if 3 * unique_identity_final < final_identity_minimum:
        raise ValueError("Unique canonical-final pool cannot reach dose under three total source occurrences")
    source_occurrences = Counter()

    def flush():
        nonlocal tokens, masks, segments, row_ids
        n = len(tokens)
        packed.append(
            {
                "input_ids": tokens + [128001] * (SEQUENCE_LENGTH - n),
                "target_weights": masks + [0.0] * (SEQUENCE_LENGTH - n),
                "segment_ids": segments + [-1] * (SEQUENCE_LENGTH - n),
            }
        )
        metadata.append({"sequence": len(packed) - 1, "valid_positions": n, "ids": row_ids})
        tokens, masks, segments, row_ids = [], [], [], []

    while True:
        share = total["identity"] / max(1, total.total())
        source = "identity" if share < identity_share else "replay"
        stream = streams[source]
        row = stream.next_row()
        ids, mask = row["input_ids"], row["assistant_masks"]
        if len(ids) > SEQUENCE_LENGTH:
            raise ValueError(f"Overlength record: {row['id']}")
        if tokens and len(tokens) + len(ids) > SEQUENCE_LENGTH:
            flush()
            if len(packed) % GLOBAL_BATCH == 0:
                batch_summaries.append(
                    {
                        "completed_updates": len(packed) // GLOBAL_BATCH,
                        "identity_supervised_tokens": total["identity"],
                        "identity_final_text_tokens": final["identity"],
                        "identity_reasoning_text_tokens": reasoning["identity"],
                        "replay_reasoning_text_tokens": reasoning["replay"],
                        "all_supervised_tokens": total.total(),
                        "identity_weighted_mass": weighted["identity"],
                        "all_weighted_mass": sum(weighted.values()),
                        "family_final_text_tokens": dict(family),
                        "family_reasoning_text_tokens": dict(family_reasoning),
                        "family_raw_supervised_tokens": dict(family_raw),
                        "family_weighted_mass": dict(family_weighted),
                        "family_row_occurrences": dict(family_rows),
                        "mode_final_text_tokens": dict(mode),
                        "structural_target_tokens_by_source": dict(structural),
                        "identity_source_occurrences_total": source_occurrences.total(),
                        "identity_distinct_sources_seen": len(source_occurrences),
                        "maximum_identity_source_occurrences_across_modes": max(source_occurrences.values(), default=0),
                    }
                )
                if final["identity"] >= final_identity_minimum:
                    break
        segment = len(row_ids)
        tokens.extend(ids)
        masks.extend(float(m) * (0.1 if t in LOW_WEIGHT_IDS else 1.0) for t, m in zip(ids, mask, strict=True))
        segments.extend([segment] * len(ids))
        row_ids.append(row["id"])
        if source == "identity":
            source_occurrences[row["source_id"]] += 1
            if source_occurrences[row["source_id"]] > 3:
                raise ValueError("Identity source conversation exceeds three TOTAL occurrences across modes")
        dose = source_pool[row["id"]]
        total[source] += dose.raw
        final[source] += dose.final_text
        reasoning[source] += dose.reasoning_text
        weighted[source] += dose.weighted
        occurrences[row["id"]] += 1
        key = f"{source}/{row['family']}"
        family[key] += dose.final_text
        family_reasoning[key] += dose.reasoning_text
        family_raw[key] += dose.raw
        family_weighted[key] += dose.weighted
        family_rows[key] += 1
        mode[f"{source}/{'on' if row['enable_thinking'] else 'off'}"] += dose.final_text
        structural.update({f"{source}/{token}": count for token, count in dose.structural.items()})
        if source == "identity":
            answer_rows[row["normalized_answer"]] += 1
            answer_final[row["normalized_answer"]] += dose.final_text
        if source == "identity" and stream.offset == 0:
            epoch_steps.append(math.ceil((len(packed) + 1) / GLOBAL_BATCH))
    updates = len(packed) // GLOBAL_BATCH
    schema = pa.schema(
        [
            ("input_ids", pa.list_(pa.int32())),
            ("target_weights", pa.list_(pa.float32())),
            ("segment_ids", pa.list_(pa.int32())),
        ]
    )
    pq.write_table(pa.Table.from_pylist(packed, schema=schema), output / "packed.parquet", compression="zstd")
    (output / "sequence-rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in metadata))
    report = {
        "updates": updates,
        "batch_size": GLOBAL_BATCH,
        "sequence_length": SEQUENCE_LENGTH,
        "identity_share_target": identity_share,
        "raw_supervised_tokens": dict(total),
        "final_text_tokens": dict(final),
        "reasoning_text_tokens": dict(reasoning),
        "weighted_mass": dict(weighted),
        "realized_identity_supervised_share": total["identity"] / total.total(),
        "realized_identity_weighted_share": weighted["identity"] / sum(weighted.values()),
        "identity_minimum_excludes_structural_tokens": final_identity_minimum,
        "family_final_text_tokens": dict(family),
        "family_reasoning_text_tokens": dict(family_reasoning),
        "family_raw_supervised_tokens": dict(family_raw),
        "family_weighted_mass": dict(family_weighted),
        "family_row_occurrences": dict(family_rows),
        "mode_final_text_tokens": dict(mode),
        "structural_target_tokens_by_source": dict(structural),
        "identity_normalized_answer_occurrences": dict(answer_rows),
        "identity_normalized_answer_final_text_tokens": dict(answer_final),
        "identity_epoch_boundary_updates": epoch_steps,
        "identity_epoch_definition": (
            "one occurrence of every ORIGINAL source conversation; balanced on/off, inverted next epoch"
        ),
        "checkpoint_updates": checkpoint_cadence(updates, epoch_steps),
        "cumulative_batches": batch_summaries,
        "row_occurrences": dict(occurrences),
        "maximum_identity_rendered_row_occurrences": max(occurrences[r["id"]] for r in identity),
        "identity_source_occurrences": dict(source_occurrences),
        "maximum_identity_source_occurrences_across_modes": max(source_occurrences.values()),
        "maximum_replay_row_occurrences": max(occurrences[r["id"]] for r in replay),
        "replay_unique_final_text_tokens": sum(source_pool[r["id"]].final_text for r in replay),
        "state": "PACKED_DOSE_PROPOSAL_REQUIRES_NATIVE_ITERATOR_AND_SCIENCE_REVIEW",
    }
    (output / "exposure.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
