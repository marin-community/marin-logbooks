# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize a run's trial results: outcome counts, score, timing."""

import collections
import glob
import json
import sys

rows = []
for path in sorted(glob.glob(f"{sys.argv[1]}/*.json")):
    with open(path) as source:
        d = json.load(source)
    error = d.get("exception_info") or {}
    rewards = (d.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    kind = error.get("exception_type") or "completed"
    message = error.get("exception_message") or ""
    if kind == "DaytonaValidationError" and "quota" in message.lower():
        kind = "snapshot quota"
    rows.append((d["task_name"], kind, reward, d.get("started_at", ""), d.get("finished_at") or ""))

outcome = collections.Counter()
for _, kind, reward, _, _ in rows:
    scored = "solved" if reward == 1.0 else "unsolved" if reward is not None else "unscored"
    outcome[(scored, kind)] += 1
for key, count in sorted(outcome.items(), key=lambda kv: -kv[1]):
    print(f"{count:4d}  {key[0]:9s} {key[1]}")
scored = [r for r in rows if r[2] is not None]
solved = [r for r in scored if r[2] == 1.0]
print(f"trials={len(rows)} scored={len(scored)} solved={len(solved)}")
quota = sorted(r[3] for r in rows if r[1] == "snapshot quota")
if quota:
    print("first quota failure started", quota[0][11:19], "last", quota[-1][11:19])
print("last scored trial finished", max((r[4] for r in scored), default="")[11:19])
if len(sys.argv) > 2:
    with open(sys.argv[2], "w") as sink:
        json.dump({r[0]: r[2] for r in rows}, sink, indent=1)
