# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Does the base model's plain-text close predict where the SFT replay closes? (marin-community/marin#10400)

Joins ``vllm_logprobs/base_tag_logprobs.py`` output (base-model p of ``</think>`` at fixed positions of the 10.05
probe trajectories) to per-trajectory dumps of the in-training probe (``reclose/<run>/step-N.jsonl``). For each
replay step it reports Spearman correlations computed within group x lead-in cells and averaged by cell size:

* ``first``: base log p(close) at the first close vs the replay's log ``p_first_close``;
* ``reclose``: the same at 10.05's own re-close vs log ``p_reclose``;
* ``hazard``: base hazard over the sampled continuation positions vs the replay's log ``hazard``.

``p(close)`` sums ``</think>`` and ``\\n</think>``. The ``control`` columns use ``\\n</assistant>`` (end of turn) in
place of the think close; ``partial`` is the think correlation after regressing the control out of both ranks,
i.e. what the think close predicts beyond a generic end-of-segment signal.

Usage: base_tag_compare.py BASE_JSON NAME=STEP_JSONL [NAME=STEP_JSONL ...]
"""

import json
import sys
from collections import defaultdict

import numpy as np

FLOOR = 1e-12
CONTINUATION_KINDS = ("after_close", "candidate", "mid")


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    out = np.empty(len(x))
    out[order] = np.arange(len(x))
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    rx, ry, rz = ranks(x), ranks(y), ranks(z)
    design = np.stack([rz, np.ones_like(rz)], axis=1)

    def residual(v: np.ndarray) -> np.ndarray:
        return v - design @ np.linalg.lstsq(design, v, rcond=None)[0]

    return float(np.corrcoef(residual(rx), residual(ry))[0, 1])


def closer_p(q: dict, names: tuple[str, ...]) -> float:
    return float(sum(np.exp(q["logp"][n]) for n in names))


def base_features(row: dict, names: tuple[str, ...]) -> dict:
    by_kind = defaultdict(list)
    for q in row["queries"]:
        by_kind[q["kind"]].append(closer_p(q, names))
    continuation = [p for k in CONTINUATION_KINDS for p in by_kind[k]]
    return {
        "first": np.log(by_kind["first_close"][0] + FLOOR) if by_kind["first_close"] else None,
        "reclose": np.log(by_kind["reclose"][0] + FLOOR) if by_kind["reclose"] else None,
        "hazard": np.log(1.0 - np.prod(1.0 - np.clip(continuation, 0, 1 - 1e-9)) + FLOOR) if continuation else None,
    }


def replay_features(row: dict) -> dict:
    return {
        "first": np.log(row["p_first_close"] + FLOOR),
        "reclose": np.log(row["p_reclose"] + FLOOR) if row["p_reclose"] is not None else None,
        "hazard": np.log(row["hazard"] + FLOOR),
    }


def key(row: dict) -> tuple:
    return row["group"], row["id"], row["sample"], row["bridge"]


def cell(row: dict) -> str:
    return f"{row['group']}/{'lead_in' if row['bridge'] else 'no_lead_in'}"


def within_cell(pairs: list[tuple[str, float, float, float]], fn) -> tuple[float, int]:
    """Cell-size-weighted mean of ``fn`` over cells with at least 20 trajectories."""
    by_cell = defaultdict(list)
    for c, *values in pairs:
        by_cell[c].append(values)
    total, weight = 0.0, 0
    for values in by_cell.values():
        if len(values) < 20:
            continue
        arr = np.array(values)
        total += fn(*arr.T) * len(arr)
        weight += len(arr)
    return (total / weight if weight else float("nan")), weight


def main() -> None:
    with open(sys.argv[1]) as handle:
        base = {key(r): r for r in json.load(handle)["rows"]}
    think = {k: base_features(r, ("think", "think_nl")) for k, r in base.items()}
    control = {k: base_features(r, ("assistant_nl",)) for k, r in base.items()}
    summary = {}
    print(f"{'run':<22}{'metric':<9}{'n':>6}{'think':>8}{'control':>9}{'partial':>9}")
    for spec in sys.argv[2:]:
        name, path = spec.split("=", 1)
        with open(path) as handle:
            rows = [json.loads(line) for line in handle]
        for metric in ("first", "reclose", "hazard"):
            pairs = []
            for r in rows:
                k = key(r)
                if k not in base:
                    continue
                b, c, s = think[k][metric], control[k][metric], replay_features(r)[metric]
                if b is None or c is None or s is None:
                    continue
                pairs.append((cell(r), b, s, c))
            rho, n = within_cell(pairs, lambda b, s, c: spearman(b, s))
            rho_control, _ = within_cell(pairs, lambda b, s, c: spearman(c, s))
            rho_partial, _ = within_cell(pairs, partial_spearman)
            summary[f"{name}/{metric}"] = {"n": n, "think": rho, "control": rho_control, "partial": rho_partial}
            print(f"{name:<22}{metric:<9}{n:>6}{rho:>8.3f}{rho_control:>9.3f}{rho_partial:>9.3f}")

    print("\nbase p(</think>) by position kind, median over trajectories")
    by = defaultdict(list)
    for r in base.values():
        for q in r["queries"]:
            by[(cell(r), q["kind"])].append(closer_p(q, ("think", "think_nl")))
    for (c, kind), values in sorted(by.items()):
        print(f"  {c:<26}{kind:<12}n={len(values):<6}median={np.median(values):.2e}  mean={np.mean(values):.2e}")
    with open(sys.argv[1].rsplit(".", 1)[0] + ".compare.json", "w") as handle:
        json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    main()
