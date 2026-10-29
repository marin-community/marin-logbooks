# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repack B-traces using the exact finite B-base occurrence ledger.

This module never selects sources or admits rationales. An external, pinned
semantic disposition supplies the accepted ON replacements. All other rows keep
their original tokens, masks and occurrences.
"""

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.post_training.identity_path_b.data import LOW_WEIGHT_IDS
from experiments.post_training.identity_path_b.exposure import (
    GLOBAL_BATCH,
    SEQUENCE_LENGTH,
    TargetDose,
    checkpoint_cadence,
    target_counts,
)


@dataclass(frozen=True)
class ScheduledRow:
    row_id: str
    source_id: str
    family: str
    enable_thinking: bool
    input_ids: np.ndarray
    assistant_masks: np.ndarray
    dose: TargetDose


def compact_row(row: dict) -> ScheduledRow:
    """Keep only the token arrays and provenance needed by finite packing."""
    return ScheduledRow(
        row_id=row["id"],
        source_id=row["source_id"],
        family=row["family"],
        enable_thinking=row["enable_thinking"],
        input_ids=np.asarray(row["input_ids"], dtype=np.int32),
        assistant_masks=np.asarray(row["assistant_masks"], dtype=np.bool_),
        dose=target_counts(row),
    )


def checked_replacement(base: dict, replacement: dict) -> ScheduledRow:
    """Refuse changed context, final token IDs, masks or source provenance."""
    provenance = ("id", "source_id", "family", "enable_thinking", "target_start", "normalized_answer")
    if not base["enable_thinking"] or any(base[key] != replacement[key] for key in provenance):
        raise ValueError("Trace replacement changed source/mode provenance")
    start = base["target_start"]
    if base["input_ids"][start : start + 2] != [128002, 128003]:
        raise ValueError("Trace authority must be the B-base empty ON row")
    ids = replacement["input_ids"]
    target = ids[start:]
    if target.count(128002) != 1 or target.count(128003) != 1 or target[0] != 128002:
        raise ValueError("Trace replacement lacks one native bounded ON span")
    boundary = target.index(128003)
    if not 8 <= boundary - 1 <= 96 or any(not 0 <= token < 128000 for token in target[1:boundary]):
        raise ValueError("Trace replacement has invalid rationale tokens")
    if ids[:start] != base["input_ids"][:start] or target[boundary + 1 :] != base["input_ids"][start + 2 :]:
        raise ValueError("Trace replacement changed prompt or canonical final IDs")
    if replacement["assistant_masks"] != [0] * start + [1] * len(target):
        raise ValueError("Trace replacement changed final-only supervision")
    if replacement["messages"][:-1] != base["messages"][:-1]:
        raise ValueError("Trace replacement changed conversation context")
    final = base["messages"][-1]["content"].removeprefix("<|start_think|><|end_think|>")
    if replacement["messages"][-1]["content"] != final:
        raise ValueError("Trace replacement changed canonical final text")
    if len(ids) > SEQUENCE_LENGTH:
        raise ValueError("Trace replacement exceeds the native packed sequence length")
    return compact_row(replacement)


def load_variant(
    identity: Path,
    replay: Path,
    drafts: list[Path],
    eligible: set[str],
    output_identity: Path,
    *,
    identity_sha256: str,
    replay_sha256: str,
) -> tuple[dict[str, ScheduledRow], dict]:
    """Copy OFF/fallback line bytes exactly; replace only approved ON drafts.

    The caller supplies the immutable external eligibility decision. This does
    not upgrade a surface/native draft to semantic acceptance.
    """
    selected = {}
    for path in drafts:
        with path.open("rb") as stream:
            for line in stream:
                draft = json.loads(line)
                source = draft["source_id"]
                if source not in eligible:
                    continue
                if source in selected:
                    raise ValueError("A source has multiple draft rationales; best-of selection is forbidden")
                selected[source] = line
    if selected.keys() != eligible:
        raise ValueError("Eligible source has no unique native-audited ON draft")
    rows, seen = {}, set()
    identity_digest, output_digest, replay_digest = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    counts = Counter()
    with identity.open("rb") as stream, output_identity.open("xb") as output:
        for line in stream:
            identity_digest.update(line)
            base = json.loads(line)
            source = base["source_id"]
            if base["enable_thinking"] and source in eligible:
                rendered = selected.pop(source)
                compact = checked_replacement(base, json.loads(rendered))
                counts["on_trace_replacements"] += 1
            else:
                rendered, compact = line, compact_row(base)
                counts["on_exact_empty_fallback" if base["enable_thinking"] else "off_exact_unchanged"] += 1
            if compact.row_id in rows:
                raise ValueError("Duplicate identity rendered-row ID")
            rows[compact.row_id] = compact
            seen.add(source)
            output.write(rendered)
            output_digest.update(rendered)
    if identity_digest.hexdigest() != identity_sha256 or selected:
        raise ValueError("Identity source bytes or complete ON replacement coverage changed")
    with replay.open("rb") as stream:
        for line in stream:
            replay_digest.update(line)
            row = compact_row(json.loads(line))
            if row.row_id in rows:
                raise ValueError("Identity and replay IDs overlap")
            rows[row.row_id] = row
    if replay_digest.hexdigest() != replay_sha256:
        raise ValueError("Replay bytes differ from the fixed B-base replay pool")
    return rows, {
        "input_identity_sha256": identity_digest.hexdigest(),
        "output_identity_sha256": output_digest.hexdigest(),
        "unchanged_replay_sha256": replay_digest.hexdigest(),
        "original_identity_sources": len(seen),
        "unique_eligible_on_sources": len(eligible),
        "identity_row_dispositions": dict(counts),
        "draft_files": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in drafts],
        "state": "PROPOSED_VARIANT_EXTERNAL_ELIGIBILITY_NOT_POOL_CLEAR",
    }


def pack_schedule(rows: dict[str, ScheduledRow], ledger: Path, baseline: dict, output: Path) -> dict:
    """Preserve every ordered occurrence; pad the final batch without extra rows.

    Raw/weighted shares and step counts are measured after replacement. The
    canonical-final minima, source repeats, replay and occurrence ledger remain
    bound to B-base. A returned proposal still needs native/science review.
    """
    ledger_bytes = ledger.read_bytes()
    ledger_sha = hashlib.sha256(ledger_bytes).hexdigest()
    if ledger_sha != baseline["ordered_occurrence_ledger_sha256"]:
        raise ValueError("B-traces must use the exact B-base ordered occurrence ledger")
    schedule = [json.loads(line) for line in ledger_bytes.splitlines()]
    output.mkdir(parents=True, exist_ok=False)
    schema = pa.schema(
        [
            ("input_ids", pa.list_(pa.int32())),
            ("target_weights", pa.list_(pa.float32())),
            ("segment_ids", pa.list_(pa.int32())),
        ]
    )
    total, final, reasoning, structural = Counter(), Counter(), Counter(), Counter()
    weighted, family_weighted = defaultdict(float), defaultdict(float)
    family_final, family_raw, family_reasoning, family_rows = Counter(), Counter(), Counter(), Counter()
    mode_final, encounters, source_occurrences, rendered_occurrences = Counter(), Counter(), Counter(), Counter()
    identity_rendered = Counter()
    tokens, weights, segments, row_ids = [], [], [], []
    sequence_count, epoch_steps, summaries = 0, [], []
    identity_total = 0

    def summary() -> dict:
        return {
            "completed_updates": sequence_count // GLOBAL_BATCH,
            "identity_supervised_tokens": total["identity"],
            "identity_final_text_tokens": final["identity"],
            "identity_reasoning_text_tokens": reasoning["identity"],
            "replay_reasoning_text_tokens": reasoning["replay"],
            "all_supervised_tokens": total.total(),
            "identity_weighted_mass": weighted["identity"],
            "all_weighted_mass": sum(weighted.values()),
            "family_final_text_tokens": dict(family_final),
            "family_reasoning_text_tokens": dict(family_reasoning),
            "family_raw_supervised_tokens": dict(family_raw),
            "family_weighted_mass": dict(family_weighted),
            "family_row_occurrences": dict(family_rows),
            "mode_final_text_tokens": dict(mode_final),
            "structural_target_tokens_by_source": dict(structural),
            "identity_source_occurrences_total": identity_total,
            "identity_distinct_sources_seen": len(source_occurrences),
            "maximum_identity_source_occurrences_across_modes": max(source_occurrences.values(), default=0),
            "identity_trace_encounters": dict(encounters),
        }

    with (
        pq.ParquetWriter(output / "packed.parquet", schema, compression="zstd") as writer,
        (output / "sequence-rows.jsonl").open("w") as metadata,
    ):

        def flush() -> None:
            nonlocal tokens, weights, segments, row_ids, sequence_count
            n = len(tokens)
            packed = {
                "input_ids": tokens + [128001] * (SEQUENCE_LENGTH - n),
                "target_weights": weights + [0.0] * (SEQUENCE_LENGTH - n),
                "segment_ids": segments + [-1] * (SEQUENCE_LENGTH - n),
            }
            writer.write_table(pa.Table.from_pylist([packed], schema=schema))
            metadata.write(json.dumps({"sequence": sequence_count, "valid_positions": n, "ids": row_ids}) + "\n")
            sequence_count += 1
            tokens, weights, segments, row_ids = [], [], [], []
            if sequence_count % GLOBAL_BATCH == 0:
                summaries.append(summary())

        for index, occurrence in enumerate(schedule):
            row = rows[occurrence["row_id"]]
            if (
                occurrence["occurrence"] != index
                or occurrence["source_id"] != row.source_id
                or occurrence["family"] != row.family
                or occurrence["enable_thinking"] != row.enable_thinking
                or occurrence["canonical_final_text_tokens"] != row.dose.final_text
            ):
                raise ValueError("Row differs from its fixed B-base occurrence")
            source = occurrence["source"]
            if source not in ("identity", "replay"):
                raise ValueError("Occurrence source must retain its B-base identity/replay binding")
            n = len(row.input_ids)
            if n > SEQUENCE_LENGTH or n != len(row.assistant_masks):
                raise ValueError("Invalid packed row token/mask length")
            if tokens and len(tokens) + n > SEQUENCE_LENGTH:
                flush()
            segment = len(row_ids)
            ids, masks = row.input_ids.tolist(), row.assistant_masks.tolist()
            tokens.extend(ids)
            weights.extend(
                float(mask) * (0.1 if token in LOW_WEIGHT_IDS else 1.0) for token, mask in zip(ids, masks, strict=True)
            )
            segments.extend([segment] * n)
            row_ids.append(row.row_id)
            dose = row.dose
            total[source] += dose.raw
            final[source] += dose.final_text
            reasoning[source] += dose.reasoning_text
            weighted[source] += dose.weighted
            key = f"{source}/{row.family}"
            family_final[key] += dose.final_text
            family_raw[key] += dose.raw
            family_reasoning[key] += dose.reasoning_text
            family_weighted[key] += dose.weighted
            family_rows[key] += 1
            mode_final[f"{source}/{'on' if row.enable_thinking else 'off'}"] += dose.final_text
            structural.update({f"{source}/{token}": count for token, count in dose.structural.items()})
            rendered_occurrences[row.row_id] += 1
            if source == "identity":
                identity_rendered[row.row_id] += 1
                source_occurrences[row.source_id] += 1
                identity_total += 1
                if source_occurrences[row.source_id] > 3:
                    raise ValueError("Original identity source exceeds three total occurrences across modes")
                category = (
                    "off_unchanged"
                    if not row.enable_thinking
                    else ("on_accepted_trace" if dose.reasoning_text else "on_empty_fallback")
                )
                encounters[category] += 1
                if identity_total % baseline["identity_original_sources"] == 0:
                    epoch_steps.append(math.ceil((sequence_count + 1) / GLOBAL_BATCH))
        if tokens:
            flush()
        while sequence_count % GLOBAL_BATCH:
            flush()

    if (
        dict(final) != baseline["final_text_tokens"]
        or reasoning["replay"] != baseline["reasoning_text_tokens"]["replay"]
    ):
        raise ValueError("Fixed-ledger replacement changed canonical final or replay reasoning dose")
    if total["replay"] != baseline["raw_supervised_tokens"]["replay"]:
        raise ValueError("Fixed-ledger replacement changed replay dose")
    if dict(rendered_occurrences) != baseline["row_occurrences"]:
        raise ValueError("Fixed-ledger replacement changed rendered-row occurrences")
    if (
        dict(family_final) != baseline["family_final_text_tokens"]
        or dict(mode_final) != baseline["mode_final_text_tokens"]
        or dict(family_rows) != baseline["family_row_occurrences"]
    ):
        raise ValueError("Fixed-ledger replacement changed family/mode final-text exposure")
    updates = sequence_count // GLOBAL_BATCH
    report = {
        "experiment": "B-traces",
        "schedule_definition": "identical row/exposure schedule",
        "ordered_occurrence_ledger_sha256": ledger_sha,
        "updates": updates,
        "base_updates": baseline["updates"],
        "step_count_change": updates - baseline["updates"],
        "batch_size": GLOBAL_BATCH,
        "sequence_length": SEQUENCE_LENGTH,
        "raw_supervised_tokens": dict(total),
        "final_text_tokens": dict(final),
        "reasoning_text_tokens": dict(reasoning),
        "weighted_mass": dict(weighted),
        "realized_identity_supervised_share": total["identity"] / total.total(),
        "realized_identity_weighted_share": weighted["identity"] / sum(weighted.values()),
        "family_final_text_tokens": dict(family_final),
        "family_reasoning_text_tokens": dict(family_reasoning),
        "family_raw_supervised_tokens": dict(family_raw),
        "family_weighted_mass": dict(family_weighted),
        "family_row_occurrences": dict(family_rows),
        "mode_final_text_tokens": dict(mode_final),
        "structural_target_tokens_by_source": dict(structural),
        "identity_trace_encounters": dict(encounters),
        "identity_source_occurrences": dict(source_occurrences),
        "maximum_identity_source_occurrences_across_modes": max(source_occurrences.values(), default=0),
        "row_occurrences": dict(rendered_occurrences),
        "maximum_identity_rendered_row_occurrences": max(identity_rendered.values(), default=0),
        "identity_epoch_boundary_updates": epoch_steps,
        "checkpoint_updates": checkpoint_cadence(updates, epoch_steps),
        "cumulative_batches": summaries,
        "state": "FIXED_LEDGER_PACKED_PROPOSAL_REQUIRES_NATIVE_AND_SCIENCE_REVIEW",
    }
    (output / "occurrence-ledger.jsonl").write_bytes(ledger_bytes)
    (output / "exposure.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
