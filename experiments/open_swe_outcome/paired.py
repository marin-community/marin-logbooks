# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paired per-task comparison of Harbor SWE-bench runs from their trial result.json files.

Each argument is ``arm:label=results_root``. A task's reward in a run is taken from its scored
trial (verifier reward present); unscored trials are ignored. Arms are compared on the tasks scored
in every run, pooling seeds by averaging each task's reward over the arm's runs. Prints per-run
resolve rates, the paired arm difference, discordant-task counts and an exact two-sided sign test.
"""

import collections
import json
import math
import sys

import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3


def task_rewards(root: str) -> dict[str, float]:
    fs, _ = fsspec.core.url_to_fs(root)
    rewards: dict[str, float] = {}
    for path in fs.glob(root.split("://", 1)[1].rstrip("/") + "/harbor_jobs/*/*/result.json"):
        trial = path.rsplit("/", 2)[-2]
        if "__" not in trial or trial.startswith("harbor_"):
            continue
        reward = ((json.loads(fs.cat_file(path)).get("verifier_result") or {}).get("rewards") or {}).get("reward")
        if reward is not None:
            rewards[trial.rsplit("__", 1)[0]] = max(rewards.get(trial.rsplit("__", 1)[0], 0.0), float(reward))
    return rewards


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> None:
    configure_coreweave_s3()
    runs: dict[str, dict[str, float]] = {}
    arm_of: dict[str, str] = {}
    for arg in sys.argv[1:]:
        name, root = arg.split("=", 1)
        arm, label = name.split(":", 1)
        runs[label] = task_rewards(root)
        arm_of[label] = arm
    for label, rewards in runs.items():
        solved = sum(rewards.values())
        rate = solved / max(len(rewards), 1)
        print(f"{label:12s} arm={arm_of[label]:7s} scored={len(rewards):3d} solved={solved:3.0f} rate={rate:.3f}")
    common = sorted(set.intersection(*(set(r) for r in runs.values())))
    arms = sorted(set(arm_of.values()))
    assert len(arms) == 2, f"expected two arms, got {arms}"
    by_arm = collections.defaultdict(list)
    for label, arm in arm_of.items():
        by_arm[arm].append(label)
    mean = {arm: {t: sum(runs[l][t] for l in labels) / len(labels) for t in common} for arm, labels in by_arm.items()}
    a, b = arms
    wins = sum(mean[a][t] > mean[b][t] for t in common)
    losses = sum(mean[a][t] < mean[b][t] for t in common)
    diff = sum(mean[a][t] - mean[b][t] for t in common)
    print(f"tasks scored in all runs: {len(common)}")
    for arm in arms:
        total = sum(mean[arm].values())
        print(f"  {arm:7s} mean solved per task {total / max(len(common), 1):.3f} (total {total:.1f})")
    p_value = sign_test(wins, losses)
    print(f"  {a} - {b}: {diff:+.1f} tasks; {a} better on {wins}, {b} better on {losses}; sign test p={p_value:.3f}")
    solved_any = [t for t in common if any(mean[arm][t] > 0 for arm in arms)]
    print(f"  tasks solved by any run: {len(solved_any)}: {', '.join(solved_any)}")


if __name__ == "__main__":
    main()
