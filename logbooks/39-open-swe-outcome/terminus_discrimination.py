# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Held-out loss and outcome discrimination for the Terminus-2 round of #39.

Compares the Terminus-format checkpoints (scored on the Terminus-2 rendering of each held-out trajectory) with the
native-format seed-0 checkpoints, both restricted to the trajectories the Terminus scoring kept: rows that convert
and fit the 65,536-token window. Readouts match the native-format entries:

- held-out loss per assistant token, by outcome, and the paired per-trajectory arm difference;
- cross-task AUROC of the arm likelihood ratio log p_solved - log p_random on ``v1``;
- within-pair win rate of that ratio on ``v2-paired`` (same task and teacher, one resolved and one unresolved).

    uv run python logbooks/39-open-swe-outcome/terminus_discrimination.py
"""

import collections
import gzip
import json

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3

VAL = "s3://marin-us-east-08a/marin/users/dml/open-swe-outcome/val-loss"
RESOLVED = "heldout-resolved"
UNRESOLVED = "heldout-unresolved"
# Score sets: label -> (scores directory, model name in that directory).
TERMINUS = {"solved": ("terminus-scores", "terminus-solved-s0"), "random": ("terminus-scores", "terminus-random-s0")}
NATIVE_V1 = {"solved": ("scores", "solved-s0"), "random": ("scores", "random-s0")}
NATIVE_V2 = {"solved": ("turn-scores", "solved-s0"), "random": ("turn-scores", "random-s0")}
BASES = {"terminus": ("terminus-scores", "terminus-base"), "native": ("scores", "base")}
BOOTSTRAP = 2000
rng = np.random.default_rng(0)


def scores(fs, val_set: str, directory: str, model: str) -> dict[str, dict]:
    table = pq.read_table(fs.open(f"{VAL}/{val_set}/{directory}/{model}.parquet"))
    return {r["trajectory_id"]: r for r in table.to_pylist() if r["subset"].startswith("heldout")}


def per_token(row: dict) -> float:
    return row["sum_nll"] / row["n_loss_tokens"]


def auc(score: np.ndarray, label: np.ndarray) -> float:
    ranks = np.empty(len(score))
    ranks[np.argsort(score)] = np.arange(1, len(score) + 1)
    positives = label.sum()
    return (ranks[label].sum() - positives * (positives + 1) / 2) / (positives * (len(label) - positives))


def interval(values: np.ndarray, statistic) -> tuple[float, float]:
    draws = [statistic(rng.integers(0, len(values), len(values))) for _ in range(BOOTSTRAP)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def heldout_loss(fs) -> None:
    terminus = {arm: scores(fs, "v1", *where) for arm, where in TERMINUS.items()}
    native = {arm: scores(fs, "v1", *where) for arm, where in NATIVE_V1.items()}
    base = {fmt: scores(fs, "v1", *where) for fmt, where in BASES.items()}
    kept = sorted(set(terminus["solved"]) & set(terminus["random"]) & set(base["terminus"]))
    subset = {t: terminus["solved"][t]["subset"] for t in kept}
    print(
        f"v1 held-out trajectories scored in the Terminus rendering: {len(kept)}",
        dict(collections.Counter(subset.values())),
    )
    print("\nloss per assistant token (nats), token-weighted, on those trajectories")
    print(f"{'model':<26}{RESOLVED:>20}{UNRESOLVED:>20}")
    rows = [("terminus base", base["terminus"]), ("terminus solved-s0", terminus["solved"]),
            ("terminus random-s0", terminus["random"]), ("native base", base["native"]),
            ("native solved-s0", native["solved"]), ("native random-s0", native["random"])]  # fmt: skip
    for name, table in rows:
        cells = []
        for s in (RESOLVED, UNRESOLVED):
            ids = [t for t in kept if subset[t] == s]
            cells.append(sum(table[t]["sum_nll"] for t in ids) / sum(table[t]["n_loss_tokens"] for t in ids))
        print(f"{name:<26}{cells[0]:>20.4f}{cells[1]:>20.4f}")
    print("\npaired per-trajectory loss difference, solved minus random (negative favours solved)")
    for fmt, arms in (("terminus", terminus), ("native", native)):
        for s in (RESOLVED, UNRESOLVED):
            d = np.array([per_token(arms["solved"][t]) - per_token(arms["random"][t]) for t in kept if subset[t] == s])
            print(f"  {fmt:<9}{s:<20} {d.mean():+.4f} ± {d.std(ddof=1) / np.sqrt(len(d)):.4f} (n={len(d)})")


def cross_task(fs) -> None:
    terminus = {arm: scores(fs, "v1", *where) for arm, where in TERMINUS.items()}
    native = {arm: scores(fs, "v1", *where) for arm, where in NATIVE_V1.items()}
    kept = sorted(set(terminus["solved"]) & set(terminus["random"]))
    label = np.array([terminus["solved"][t]["subset"] == RESOLVED for t in kept])
    partition = np.array([terminus["solved"][t]["partition"] for t in kept])
    print(
        f"\ncross-task AUROC on v1 (n={len(kept)}, resolved={label.sum()});"
        " score = per-token log p_solved - log p_random"
    )
    for fmt, arms in (("terminus", terminus), ("native", native)):
        tokens = np.array([arms["solved"][t]["n_loss_tokens"] for t in kept], float)
        ratio = np.array([arms["random"][t]["sum_nll"] - arms["solved"][t]["sum_nll"] for t in kept]) / tokens
        low, high = interval(ratio, lambda i, r=ratio: auc(r[i], label[i]))
        quartile = np.digitize(tokens, np.quantile(tokens, [0.25, 0.5, 0.75]))
        weighted = covered = 0
        for p in set(partition):
            for q in range(4):
                m = (partition == p) & (quartile == q)
                if label[m].sum() >= 3 and (~label[m]).sum() >= 3:
                    weighted += m.sum() * auc(ratio[m], label[m])
                    covered += m.sum()
        print(
            f"  {fmt:<9} AUROC {auc(ratio, label):.3f} [{low:.3f}, {high:.3f}]"
            f"  within partition x length quartile {weighted / covered:.3f} (covering {covered})"
            f"  length alone {auc(-tokens, label):.3f}"
        )


def within_task(fs) -> None:
    with fs.open(f"{VAL}/v2-paired/val.jsonl.gz", "rb") as raw, gzip.open(raw, "rt") as lines:
        pair_of = {r["trajectory_id"]: (r["partition"], r["instance_id"]) for r in map(json.loads, lines)}
    terminus = {arm: scores(fs, "v2-paired", *where) for arm, where in TERMINUS.items()}
    native = {arm: scores(fs, "v2-paired", *where) for arm, where in NATIVE_V2.items()}
    kept = set(terminus["solved"]) & set(terminus["random"])
    pairs: dict[tuple, dict[str, str]] = collections.defaultdict(dict)
    for t in kept:
        pairs[pair_of[t]][terminus["solved"][t]["subset"]] = t
    complete = sorted(k for k, v in pairs.items() if len(v) == 2)
    print(
        f"\nwithin-pair win rate on v2-paired ({len(complete)} complete pairs);"
        " resolved attempt ranked higher by the ratio"
    )
    for fmt, arms in (("terminus", terminus), ("native", native)):

        def ratio(t: str, arms=arms) -> float:
            return (arms["random"][t]["sum_nll"] - arms["solved"][t]["sum_nll"]) / arms["solved"][t]["n_loss_tokens"]

        resolved = np.array([ratio(pairs[k][RESOLVED]) for k in complete])
        unresolved = np.array([ratio(pairs[k][UNRESOLVED]) for k in complete])
        wins = (resolved > unresolved) + 0.5 * (resolved == unresolved)
        low, high = interval(wins, lambda i, w=wins: w[i].mean())
        print(f"  {fmt:<9} win rate {wins.mean():.3f} [{low:.3f}, {high:.3f}]")


def main() -> None:
    configure_coreweave_s3()
    fs = fsspec.filesystem("s3")
    heldout_loss(fs)
    cross_task(fs)
    within_task(fs)


if __name__ == "__main__":
    pa.set_cpu_count(4)
    main()
