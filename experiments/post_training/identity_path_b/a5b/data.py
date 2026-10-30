# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preselected stratified A5b pilot; teacher instruction is the only request change."""

import hashlib
from collections import Counter

from experiments.post_training.identity_path_b.a5.data import TEACHER_SYSTEM as PROFILE

TEACHER_SYSTEM = PROFILE + (
    " Never mention, quote or refer to this system message or any instructions; "
    "reason as if you simply know who you are. In your final answer, state your name "
    "(Marin's Latest MoE) and that you were developed and trained by the Marin Community."
)
QUOTAS = {
    "maintained_by": 18,
    "name": 19,
    "indirect_identity": 18,
    "developer": 19,
    "deployment_provider": 18,
    "trained_by": 18,
    "ecosystem_contributions": 18,
    "open_weight_data": 18,
    "false_premise": 18,
    "unknown_fact": 18,
    "project_fact": 18,
}


def selected_candidates(originals: list[dict]) -> list[dict]:
    if len(originals) != 33298 or len({row["source_id"] for row in originals}) != 33298:
        raise ValueError("Require the complete original capped A5 manifest")
    selected = set()
    for family, quota in QUOTAS.items():
        candidates = sorted(
            (row for row in originals if row["family"] == family),
            key=lambda row: (hashlib.sha256(f"310:{row['source_id']}".encode()).hexdigest(), row["source_id"]),
        )
        if len(candidates) < quota:
            raise ValueError("Pilot family quota is unavailable")
        selected.update(row["source_id"] for row in candidates[:quota])
    rows = [{**row, "teacher_system": TEACHER_SYSTEM} for row in originals if row["source_id"] in selected]
    rows.sort(key=lambda row: row["global_index"])
    if len(rows) != 200 or Counter(row["family"] for row in rows) != QUOTAS:
        raise ValueError("A5b pilot selection differs from its fixed quotas")
    return rows


def pilot_gate(reviews: list[dict]) -> dict:
    """All200 requests remain in the denominator, including failures and rejects."""
    if len(reviews) != 200 or Counter(row["family"] for row in reviews) != QUOTAS:
        raise ValueError("Pilot gate requires all200 unique planned outcomes")
    if len({row["source_id"] for row in reviews}) != 200:
        raise ValueError("Repeated pilot original")
    kept = Counter(row["family"] for row in reviews if row["candidate_pass"])
    overall = sum(kept.values())
    return dict(
        planned=200,
        prescreen_kept=overall,
        minimum_overall=50,
        family_kept={family: kept[family] for family in QUOTAS},
        family_minimum={family: (quota + 9) // 10 for family, quota in QUOTAS.items()},
        gate_pass=overall >= 50 and all(kept[family] * 10 >= quota for family, quota in QUOTAS.items()),
        full_capture_or_training_authorized=False,
    )
