# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the cleared B-traces pool to B-base's exact finite row schedules."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from experiments.post_training.identity_path_b.base import ARMS
from experiments.post_training.identity_path_b.exposure import GLOBAL_BATCH, SEQUENCE_LENGTH, checkpoint_cadence
from experiments.post_training.identity_path_b.trace_schedule import ScheduledRow, compact_row, pack_schedule

TRACE_BINDING = {
    "experiment": "B-traces",
    "identity_pool_sha256": "566fc1800e5b3b8115312e5bdbaa135cac08296836f09e37c74c57565ac69f9f",
    "identity_parent_pool_sha256": "0d0fd663593786ea356e843eb0a64d684778174c9585d25110b142b8c5f84930",
    "replay_pool_sha256": "affce0a332e3a07bbfe0bba7690572112a83fcec3ba72457507602e630f8b91a",
    "pool_science_verdict_sha256": "c940895afb5308364dd81b9e4d83f1028b1c19c9cf2a4617be7ea5db2acf3430",
    "all_source_dispositions_sha256": "7d85fa9f57e9b47535f9c65dafd4b3712a38c661c0fedacd8e339ee4de00263d",
    "identity_original_sources": 40978,
    "identity_rendered_rows": 81956,
    "identity_traced_on_sources": 27161,
    "identity_empty_fallback_on_sources": 13817,
    "identity_unique_final_text_tokens": 843552,
    "identity_unique_reasoning_text_tokens": 842757,
    "schedule_definition": "identical row/exposure schedule",
}
ARM_BINDINGS = {
    "10": {
        "baseline_exposure_sha256": "420669f1598d7c912d24c8ddcff179950c2a90efce917a3da2b0aef5419399d9",
        "ordered_occurrence_ledger_sha256": "7c1a1b337e0dacd4f14d7a5ea80109ff4722a49d59e1606e87f03dfab1605226",
        "final_text": 501320,
        "reasoning_text": 248633,
        "off": 12171,
        "traced_on": 8047,
        "empty_on": 4132,
        "maximum_source_occurrences": 1,
    },
    "25": {
        "baseline_exposure_sha256": "8f62ba2c2c3cbb8a164a79ad9dc9367e07a5fb93cf871369846dcf8c502bced5",
        "ordered_occurrence_ledger_sha256": "a5fc778eb4c82f2fc203ef0f0d67f52b990e90454ceec0a59ac21effce046c39",
        "final_text": 1506456,
        "reasoning_text": 753086,
        "off": 36632,
        "traced_on": 24282,
        "empty_on": 12302,
        "maximum_source_occurrences": 2,
    },
}


def check_trace_exposure(exposure: dict, arm: str) -> None:
    """Admit this pool and fixed final/rationale dose, keeping B-base separate."""
    if any(exposure.get(key) != value for key, value in TRACE_BINDING.items()):
        raise ValueError("B-traces exposure is not bound to its cleared logical pool")
    expected = ARM_BINDINGS[arm]
    if any(
        exposure.get(key) != expected[key]
        for key in (
            "baseline_exposure_sha256",
            "ordered_occurrence_ledger_sha256",
        )
    ):
        raise ValueError("B-traces must preserve the exact B-base arm and occurrence ledger")
    share, minimum = ARMS[arm]
    if (
        exposure["identity_share_target"] != share
        or exposure["identity_minimum_excludes_structural_tokens"] != minimum
        or exposure["batch_size"] != GLOBAL_BATCH
        or exposure["sequence_length"] != SEQUENCE_LENGTH
    ):
        raise ValueError("B-traces changed the arm identity-final dose or packed dimensions")
    if (
        exposure["final_text_tokens"]["identity"] != expected["final_text"]
        or exposure["final_text_tokens"]["identity"] < minimum
        or exposure["reasoning_text_tokens"]["identity"] != expected["reasoning_text"]
    ):
        raise ValueError("B-traces must preserve canonical-final dose and separately report admitted rationale dose")
    if exposure["identity_trace_encounters"] != {
        "off_unchanged": expected["off"],
        "on_accepted_trace": expected["traced_on"],
        "on_empty_fallback": expected["empty_on"],
    }:
        raise ValueError("B-traces traced/fallback/OFF exposure differs from its fixed schedule")
    if (
        exposure["maximum_identity_source_occurrences_across_modes"] != expected["maximum_source_occurrences"]
        or exposure["maximum_identity_rendered_row_occurrences"] != 1
    ):
        raise ValueError("B-traces changed original-source or rendered-row occurrence caps")
    if (
        exposure["checkpoint_updates"]
        != checkpoint_cadence(exposure["updates"], exposure["identity_epoch_boundary_updates"])
        or len(exposure["cumulative_batches"]) != exposure["updates"]
    ):
        raise ValueError("B-traces checkpoint cadence or cumulative update accounting differs")
    last = exposure["cumulative_batches"][-1]
    if (
        last["identity_final_text_tokens"] != expected["final_text"]
        or last["identity_reasoning_text_tokens"] != expected["reasoning_text"]
        or last["all_supervised_tokens"] != sum(exposure["raw_supervised_tokens"].values())
    ):
        raise ValueError("B-traces last consumed batch differs from its reported final/rationale exposure")


def load_cleared_rows(identity: Path, replay: Path) -> dict[str, ScheduledRow]:
    """Read exact pool bytes into compact host arrays without selecting sources."""
    rows = {}
    for path, expected in (
        (identity, TRACE_BINDING["identity_pool_sha256"]),
        (replay, TRACE_BINDING["replay_pool_sha256"]),
    ):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                row = compact_row(json.loads(line))
                if row.row_id in rows:
                    raise ValueError("Cleared identity/replay rendered IDs must be disjoint and unique")
                rows[row.row_id] = row
        if digest.hexdigest() != expected:
            raise ValueError("B-traces logical identity or replay pool bytes differ from the cleared source")
    return rows


def family_coverage(rows: dict[str, ScheduledRow], ledger: Path) -> dict:
    """Measure traced/fallback/OFF occurrences and final-token coverage by family."""
    coverage = defaultdict(Counter)
    with ledger.open("rb") as stream:
        for line in stream:
            occurrence = json.loads(line)
            if occurrence["source"] != "identity":
                continue
            row = rows[occurrence["row_id"]]
            category = (
                "off_unchanged"
                if not row.enable_thinking
                else ("on_accepted_trace" if row.dose.reasoning_text else "on_empty_fallback")
            )
            result = coverage[row.family]
            result[category] += 1
            result[f"{category}_final_text_tokens"] += row.dose.final_text
            result[f"{category}_reasoning_text_tokens"] += row.dose.reasoning_text
    return dict(coverage)


def build_arm(rows: dict[str, ScheduledRow], arm: str, parent: Path, output: Path) -> dict:
    """Repack one reviewed finite ledger, then bind the resulting measured exposure."""
    baseline_raw = (parent / "exposure.json").read_bytes()
    if hashlib.sha256(baseline_raw).hexdigest() != ARM_BINDINGS[arm]["baseline_exposure_sha256"]:
        raise ValueError("B-traces baseline must be the exact cleared B-base exposure")
    baseline = json.loads(baseline_raw)
    ledger = parent / "occurrence-ledger.jsonl"
    report = pack_schedule(rows, ledger, baseline, output)
    report.update(TRACE_BINDING)
    report.update(
        baseline_exposure_sha256=ARM_BINDINGS[arm]["baseline_exposure_sha256"],
        identity_share_target=baseline["identity_share_target"],
        identity_minimum_excludes_structural_tokens=baseline["identity_minimum_excludes_structural_tokens"],
        family_trace_coverage=family_coverage(rows, ledger),
    )
    check_trace_exposure(report, arm)
    (output / "exposure.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_cleared_rows(args.identity, args.replay)
    args.output.mkdir(parents=True, exist_ok=False)
    for arm in ARM_BINDINGS:
        report = build_arm(rows, arm, args.base_root / f"arm{arm}", args.output / f"arm{arm}")
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "updates",
                        "step_count_change",
                        "final_text_tokens",
                        "reasoning_text_tokens",
                        "checkpoint_updates",
                        "identity_on_trace_row_fraction",
                        "realized_identity_supervised_share",
                        "realized_identity_weighted_share",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
