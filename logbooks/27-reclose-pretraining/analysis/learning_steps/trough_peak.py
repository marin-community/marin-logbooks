# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trough-to-peak learning steps for closing and closing twice, per replay and turn group.

Each series is rescaled from its trough (0) to its peak (1) over the pair of steps with the largest rise, and
the steps where it first reaches 10%, 50% and 90% are reported for the close metric and the re-close metric.
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
GROUPS = {"swe": "swe", "other+tool": "tool", "other-plain": "plain"}
LEVELS = (0.1, 0.5, 0.9)
OLD_PROBE = {
    "answer (old probe)": "probe_replay-10400-base300e.csv",
    "mean (old probe)": "probe_replay-10400-meanseed.csv",
}
ESTIMATE = {"answer": "answer (est)", "mean": "mean (est)", "ltslash": "</ (est, 20 steps)"}
EXACT = {"answer": "answer (exact)", "ltslash": "</ (exact)"}

Series = dict[str, dict[int, float]]


def crossings(series: dict[int, float]) -> tuple[list[int], int, int, float, float]:
    """Steps reaching each level over the largest rise, plus the trough step, peak step and their values."""
    steps = sorted(series)
    vals = [series[s] for s in steps]
    best, trough, peak, low = -1.0, 0, 0, 0
    for j in range(len(vals)):
        if vals[j] < vals[low]:
            low = j
        if vals[j] - vals[low] > best:
            best, trough, peak = vals[j] - vals[low], low, j
    lo, hi = vals[trough], vals[peak]
    hits = [next(steps[i] for i in range(trough, peak + 1) if vals[i] >= lo + level * (hi - lo)) for level in LEVELS]
    return hits, steps[trough], steps[peak], lo, hi


def load_series() -> dict[str, tuple[Series, Series, str, str]]:
    series = {}
    for name, filename in OLD_PROBE.items():
        first, again = {}, {}
        with (ANALYSIS / filename).open() as f:
            for row in csv.DictReader(f):
                group, step = GROUPS[row["group"]], int(row["step"])
                first.setdefault(group, {})[step] = float(row["p_first_close"])
                again.setdefault(group, {})[step] = float(row["p_reclose"])
        series[name] = (first, again, "p_first_close", "p_reclose")
    data = json.loads((HERE / "page_data.json").read_text())
    for key, name in ESTIMATE.items():
        rows = data["est"][key]
        series[name] = (
            {g: {p[0]: p[1] for p in v} for g, v in rows.items()},
            {g: {p[0]: p[2] for p in v} for g, v in rows.items()},
            "P(>=1)",
            "P(>=2)",
        )
    for key, name in EXACT.items():
        rows = data["exact"][key]
        series[name] = (
            {g: {p[0]: p[1] for p in v} for g, v in rows.items()},
            {g: {p[0]: p[2] for p in v} for g, v in rows.items()},
            "P(>=1)",
            "hazard",
        )
    return series


def main() -> None:
    print(f"{'run':22} {'grp':5} {'close':13} {'10/50/90%':14} {'re-close':15} {'10/50/90%':14} {'lag at 50%':>10}")
    results = []
    for name, (close, reclose, close_metric, reclose_metric) in load_series().items():
        for group in ("swe", "tool", "plain"):
            c, *c_range = crossings(close[group])
            r, *r_range = crossings(reclose[group])
            print(
                f"{name:22} {group:5} {close_metric:13} {c!s:14} {reclose_metric:15} {r!s:14} {r[1] - c[1]:>+10}"
                f"   close s{c_range[0]}->s{c_range[1]}, re-close s{r_range[0]}->s{r_range[1]}"
            )
            results.append(
                {
                    "run": name,
                    "group": group,
                    "close_metric": close_metric,
                    "close": c,
                    "close_range": c_range,
                    "reclose_metric": reclose_metric,
                    "reclose": r,
                    "reclose_range": r_range,
                }
            )
    (HERE / "trough_peak.json").write_text(json.dumps(results))


if __name__ == "__main__":
    main()
