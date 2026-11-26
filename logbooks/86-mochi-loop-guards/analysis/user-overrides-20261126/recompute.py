# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
from datetime import UTC, datetime
from pathlib import Path

root = Path(__file__).resolve().parent
out = root / "recomputed"
out.mkdir(exist_ok=True)
now = datetime.now(UTC).isoformat()
clock = {
    "kind": "user override",
    "recorded_at": now,
    "T0": "2026-11-25T12:01:36-05:00",
    "extension_hours": 24,
    "T20": "2026-11-27T08:01:36-05:00",
    "T32": "2026-11-27T20:01:36-05:00",
    "deadline": "2026-11-28T12:01:36-05:00",
    "decision": "I3 restored; supersedes original T20 drop; original receipt retained",
    "supersedes": (
        "s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/decisions/i3-t20-budget-drop-2026.11.26.json"
    ),
}
(out / "clock-extension-user-override.json").write_text(json.dumps(clock, indent=2) + "\n")
results = []
for arm in ("stop", "rp"):
    r = json.loads((root / f"s1-{arm}-full-gate-receipt.json").read_text())
    original = json.loads(json.dumps(r))
    triggers = []
    for row in r["cohorts"]:
        raw = row["correctness_min_raw"] * 0.75
        row["original_correctness_min_raw"] = row["correctness_min_raw"]
        row["correctness_min_raw"] = raw
        row["correctness_min_integer"] = math.ceil(raw)
        row["cohort_correctness_pass"] = row["passes"] >= raw
        row["correctness_boundary_repeat_trigger"] = abs(row["passes"] - raw) <= 2
        if row["correctness_boundary_repeat_trigger"]:
            triggers.append(row["cohort"] + " correctness")
        if arm == "rp" and abs(row["action_loop_UB_count"] - row["primary_UB_cap_raw"]) <= 2:
            triggers.append(row["cohort"] + " UB")
    r["pooled_correctness_min_raw"] = 222.54 * 0.75
    r["pooled_correctness_min_integer"] = math.ceil(r["pooled_correctness_min_raw"])
    r["pooled_correctness_pass"] = r["passes"] >= r["pooled_correctness_min_raw"]
    r.pop("trigger", None)
    r.pop("repeat_scope", None)
    r["repeat_triggered"] = bool(triggers)
    r["triggers"] = triggers
    r["final_guard_verdict"] = "FAIL"
    r["failure_reasons"] = (
        ["TBLite repeat-turn share exceeds unchanged I1 cap"]
        if arm == "stop"
        else ["All three action UB counts exceed unchanged I2 caps"]
    )
    results.append({"arm": arm, "original": original, "under_user_override": r})
gate = {
    "kind": "user override",
    "recorded_at": now,
    "correctness_multiplier": 0.75,
    "loop_gates_unchanged": True,
    "repeat_trigger_uses_unrounded_thresholds": True,
    "results": results,
    "RP_repeat_decision": (
        "No preregistered trigger; cancel exact repeat; originals remain authoritative; "
        "repeat excluded from scientific verdict"
    ),
}
assert all(not x["under_user_override"]["repeat_triggered"] for x in results)
(out / "correctness-gate-user-override.json").write_text(json.dumps(gate, indent=2) + "\n")
con = {
    "kind": "user override",
    "recorded_at": now,
    "permanent": True,
    "exclude_from_every_pruning_sweep": True,
    "source": "s3://marin-us-east-08a/marin/models/mochi-sft/hero-sft-fullab-step393k-v2/step-400407/2af3f828aa-cpu-a1",
    "HF": "open-athena/Marin-Mochi-535B-A23B@sft-fullab-v2-step-400407",
    "HF_private": False,
    "private_conversion_blocker": (
        "HF403: Private storage limit reached for org open-athena; org plan/capacity must be resolved"
    ),
    "new_S1_weights": False,
    "new_S2_weights_policy": (
        "Export both arms to PRIVATE open-athena repos with OpenMDW1.1 LICENSE, "
        "required model-card warnings and generation config via in-region CPU job; "
        "conserve regional exports permanently"
    ),
    "hash_verification": "pending regional verification",
}
(out / "checkpoint-conservation-user-override.json").write_text(json.dumps(con, indent=2) + "\n")
print(
    json.dumps(
        {
            "clock": clock,
            "raw_correctness": [40.65 * 0.75, 56 * 0.75, 104.2 * 0.75, 222.54 * 0.75],
            "stop": "FAIL loop gate only",
            "rp": "FAIL loop gates only",
            "repeat_required": False,
        },
        indent=2,
    )
)
