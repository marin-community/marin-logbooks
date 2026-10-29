# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from types import SimpleNamespace

from experiments.post_training.identity_path_b.refusal_cap import capped_refilled_schedule


def test_refill_preserves_replay_and_retained_order_without_repeating_refusal_or_exceeding_three():
    rows = {}
    for source, family in [("a", "name"), ("b", "trained_by"), ("r", "unknown_fact")]:
        for mode in (False, True):
            row_id = f"{source}-{mode}"
            rows[row_id] = SimpleNamespace(
                row_id=row_id, source_id=source, family=family, enable_thinking=mode, dose=SimpleNamespace(final_text=10)
            )

    def occurrence(source, mode=False):
        return dict(
            source="identity",
            source_id=source,
            row_id=f"{source}-{mode}",
            family="unknown_fact" if source in {"r", "d"} else "name",
            enable_thinking=mode,
            canonical_final_text_tokens=10,
        )

    original = [
        occurrence("a"),
        occurrence("d"),
        dict(
            source="replay",
            source_id="p",
            row_id="replay",
            family="chat",
            enable_thinking=False,
            canonical_final_text_tokens=15,
        ),
        occurrence("a", True),
        occurrence("r"),
        occurrence("d", True),
    ]
    for i, r in enumerate(original):
        r["occurrence"] = i
    schedule, proof = capped_refilled_schedule(original, rows, {"d"}, 65)
    assert proof["realized_final_tokens"] >= 65
    assert proof["maximum_source_occurrences"] == 3
    assert all(r["source_id"] not in {"r", "d"} for r in proof["extra_occurrences"])
    counts = Counter(r["source_id"] for r in schedule if r["source"] == "identity")
    assert counts["r"] == 1 and counts["d"] == 0 and max(counts.values()) <= 3
    before = [{k: v for k, v in r.items() if k != "occurrence"} for r in original if r["source"] == "replay"]
    after = [{k: v for k, v in r.items() if k != "occurrence"} for r in schedule if r["source"] == "replay"]
    assert before == after
    retained = [r for r, o in zip(schedule, proof["origin_map"], strict=True) if "parent_occurrence" in o]
    assert [r["row_id"] for r in retained] == [r["row_id"] for r in original if r["source_id"] != "d"]
    again, repeat_proof = capped_refilled_schedule(original, rows, {"d"}, 65)
    assert schedule == again and proof == repeat_proof
