# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact two-update diagnostic prefix, distinct from full-dose admission."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from experiments.post_training.identity_path_b.base import BASE_BINDING, check_base_exposure
from experiments.post_training.identity_path_b.storage_audit import FROZEN_HF_NAMES

SMOKE_PARENT = {
    "packed_sha256": "8067f65fee37e9f4f3805f98120e720492e0a5a96138d6eded2216b1cd31464d",
    "exposure_sha256": "420669f1598d7c912d24c8ddcff179950c2a90efce917a3da2b0aef5419399d9",
    "ordered_occurrence_ledger_sha256": "7c1a1b337e0dacd4f14d7a5ea80109ff4722a49d59e1606e87f03dfab1605226",
    "sequence_rows_sha256": "2800e1df5d5d73060793c621a4cf43570ed9aafb638af644d4016f4112351007",
    "first_two_cumulative_sha256": "4ac2db5d3210ab3b579b591b90f2ae233c03a0df2860e241c6ed7552cf1f424f",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def check_training_exposure(exposure: dict, arm: str) -> None:
    if exposure.get("experiment") != "B-base-smoke":
        check_base_exposure(exposure, arm)
        return
    if arm != "10" or any(exposure.get(k) != v for k, v in BASE_BINDING.items() if k != "experiment"):
        raise ValueError("Smoke must retain the exact B-base identity pool and arm10")
    if exposure.get("smoke_parent") != SMOKE_PARENT:
        raise ValueError("Smoke parent differs from the cleared arm10")
    if (
        exposure["updates"] != 2
        or exposure["batch_size"] != 8
        or exposure["sequence_length"] != 8192
        or exposure["checkpoint_updates"] != [1]
        or exposure["identity_minimum_excludes_structural_tokens"] != 0
        or exposure["identity_share_target"] != 0.1
    ):
        raise ValueError("Smoke requires exactly two unchanged batches and update1 diagnostics")
    cumulative = exposure["cumulative_batches"]
    if sha(json.dumps(cumulative, sort_keys=True).encode()) != SMOKE_PARENT["first_two_cumulative_sha256"]:
        raise ValueError("Smoke cumulative exposure differs from the exact two-batch prefix")
    last = cumulative[-1]
    if (
        exposure["final_text_tokens"]["identity"] != last["identity_final_text_tokens"]
        or exposure["raw_supervised_tokens"]["identity"] != last["identity_supervised_tokens"]
        or sum(exposure["raw_supervised_tokens"].values()) != last["all_supervised_tokens"]
        or exposure["reasoning_text_tokens"]["identity"] != 0
        or exposure["maximum_identity_source_occurrences_across_modes"] != 1
    ):
        raise ValueError("Smoke must report actual small exposure, not full-arm dose")


def build_smoke(parent: Path, output: Path) -> dict:
    bindings = {
        "packed.parquet": "packed_sha256",
        "exposure.json": "exposure_sha256",
        "occurrence-ledger.jsonl": "ordered_occurrence_ledger_sha256",
        "sequence-rows.jsonl": "sequence_rows_sha256",
    }
    raw = {name: (parent / name).read_bytes() for name in bindings}
    if any(sha(raw[name]) != SMOKE_PARENT[key] for name, key in bindings.items()):
        raise ValueError("Smoke inputs must be the exact cleared arm10 artifacts")
    full = json.loads(raw["exposure.json"])
    check_base_exposure(full, "10")
    cumulative = full["cumulative_batches"][:2]
    last = cumulative[-1]
    sequences = raw["sequence-rows.jsonl"].splitlines(keepends=True)[:16]
    ids = [row_id for line in sequences for row_id in json.loads(line)["ids"]]
    ledger = raw["occurrence-ledger.jsonl"].splitlines(keepends=True)[: len(ids)]
    occurrences = [json.loads(line) for line in ledger]
    if ids != [row["row_id"] for row in occurrences]:
        raise ValueError("Two-batch prefix changed the ordered occurrence schedule")
    counts = Counter(row["row_id"] for row in occurrences)
    sources = Counter(row["source_id"] for row in occurrences if row["source"] == "identity")
    report = {
        **BASE_BINDING,
        "experiment": "B-base-smoke",
        "smoke_parent": SMOKE_PARENT,
        "updates": 2,
        "batch_size": 8,
        "sequence_length": 8192,
        "checkpoint_updates": [1],
        "identity_share_target": 0.1,
        "identity_minimum_excludes_structural_tokens": 0,
        "parent_full_arm_final_text_minimum": 500000,
        "state": "DIAGNOSTIC_PREFIX_ONLY_NOT_FULL_DOSE_ADMISSION",
        "cumulative_batches": cumulative,
        "raw_supervised_tokens": {
            "identity": last["identity_supervised_tokens"],
            "replay": last["all_supervised_tokens"] - last["identity_supervised_tokens"],
        },
        "final_text_tokens": {
            kind: sum(v for k, v in last["family_final_text_tokens"].items() if k.startswith(kind + "/"))
            for kind in ("identity", "replay")
        },
        "reasoning_text_tokens": {
            "identity": last["identity_reasoning_text_tokens"],
            "replay": last["replay_reasoning_text_tokens"],
        },
        "weighted_mass": {
            "identity": last["identity_weighted_mass"],
            "replay": last["all_weighted_mass"] - last["identity_weighted_mass"],
        },
        "maximum_identity_source_occurrences_across_modes": max(sources.values()),
        "row_occurrences": dict(counts),
        "identity_source_occurrences": dict(sources),
        "ordered_occurrence_ledger_sha256": sha(b"".join(ledger)),
    }
    for key in (
        "family_final_text_tokens",
        "family_reasoning_text_tokens",
        "family_raw_supervised_tokens",
        "family_weighted_mass",
        "family_row_occurrences",
        "mode_final_text_tokens",
        "structural_target_tokens_by_source",
    ):
        report[key] = last[key]
    report["replay_pool_sha256"] = full["replay_pool_sha256"]
    report["realized_identity_supervised_share"] = last["identity_supervised_tokens"] / last["all_supervised_tokens"]
    report["realized_identity_weighted_share"] = last["identity_weighted_mass"] / last["all_weighted_mass"]
    check_training_exposure(report, "10")
    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(pq.read_table(parent / "packed.parquet").slice(0, 16), output / "packed.parquet", compression="zstd")
    (output / "sequence-rows.jsonl").write_bytes(b"".join(sequences))
    (output / "occurrence-ledger.jsonl").write_bytes(b"".join(ledger))
    (output / "exposure.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def require_smoke_success(
    *,
    training: dict,
    native_metadata: dict,
    exported_event: dict,
    storage_audit: dict,
    trend: dict,
    expected_event: dict,
    gsm_inputs: dict,
    generations: bytes,
) -> None:
    """Require real bound update/save/audited-export/trend evidence, not job success."""
    if training.get("completed_updates") != 2 or native_metadata.get("step") != 0:
        raise ValueError("Smoke did not complete two updates and save update1")
    for key in ("arm", "training_pin"):
        if training.get(key) != expected_event[key]:
            raise ValueError("Smoke training result differs from the bound event")
    if training.get("packed_sha256") != expected_event["packed_data_sha256"]:
        raise ValueError("Smoke training input differs from the bound prefix")
    checkpoints = training.get("checkpoints", [])
    if not any(
        row.get("completed_updates") == 1
        and row.get("native_checkpoint") == expected_event["native_checkpoint"]
        and row.get("exposure") == expected_event["exposure"]
        for row in checkpoints
    ):
        raise ValueError("Smoke training result lacks the committed update1 event")
    if exported_event != expected_event or trend.get("event") != expected_event:
        raise ValueError("Smoke export/trend is not the exact update1 event")
    if (
        storage_audit.get("schema") != "path_b_storage_v1"
        or storage_audit.get("frozen_native_tensors") != 104
        or storage_audit.get("trained_master_cast_tensors") != 398
        or len(storage_audit.get("tensors", {})) != 502
    ):
        raise ValueError("Smoke export lacks the complete native104/master398 storage audit")
    tensors = storage_audit["tensors"]
    if (
        storage_audit.get("source_receipt_sha256") != "af484f49ac2c1d67989495475bcc79bf7519d09281c8087321d906b67139fa97"
        or {name for name, row in tensors.items() if row.get("frozen_source")} != FROZEN_HF_NAMES
        or any(
            not isinstance(row.get("export_sha256"), str) or len(row["export_sha256"]) != 64 for row in tensors.values()
        )
    ):
        raise ValueError("Smoke export storage audit differs from the pinned source or tensor hashes")
    coverage = trend.get("identity_control_coverage", {})
    if (
        trend.get("status") != "generated_pending_semantic_read"
        or coverage.get("requested") != 66
        or coverage.get("received") != 66
        or coverage.get("errors") != 0
        or not trend.get("gsm_coverage_complete")
    ):
        raise ValueError("Smoke trend row has incomplete diagnostic coverage")
    rows = [json.loads(line) for line in generations.splitlines()]
    if (
        sha(generations) != coverage.get("generations_sha256")
        or len(rows) != 66
        or len({row["row_id"] for row in rows}) != 66
        or Counter(row.get("set") for row in rows) != {"identity": 52, "control": 14}
        or any(row.get("coverage") != "response_received" or row.get("thinking") is not False for row in rows)
    ):
        raise ValueError("Smoke trend lacks the hash-bound66 complete OFF generations")
    gsm_coverage = list(trend.get("gsm_outcome", {}).get("coverage", {}).values())
    if (
        len(gsm_coverage) != 1
        or gsm_coverage[0].get("n_scored") != 1319
        or gsm_inputs.get("count") != 1319
        or gsm_inputs.get("canonical_question_answer_jsonl_sha256")
        != "32c548f08195e19e33408b844dd7be6aa4bcae457d957bc05909ff4bd4a00595"
    ):
        raise ValueError("Smoke GSM trend lacks all1319 pinned scored inputs")
