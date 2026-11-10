# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare unlikelihood replays to the plain replay at equal closing rates.

Reads the in-loop teacher-forced probe series (one row per step: step, then
tool-using hazard, tool-using P(>=1 close), SWE hazard, SWE P(>=1 close)) and,
for each penalized step, interpolates the plain run's re-close hazard at the
same P(>=1 close). This removes the part of a hazard drop that only comes from
learning to close later.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
FIRST_STEP = 157000


def series(name: str) -> dict[int, list[float]]:
    rows = {}
    for line in (HERE / name).read_text().splitlines():
        fields = line.split()
        rows.setdefault(int(fields[1]) - FIRST_STEP, [float(fields[i]) for i in (3, 5, 7, 9)])
    return rows


def matched(plain: dict[int, list[float]], group: str, p_ge1: float) -> float:
    """Plain-run hazard at the plain run's first rise through `p_ge1`."""
    hazard_col, p_col = {"tool": (0, 1), "swe": (2, 3)}[group]
    steps = sorted(plain)
    ps = np.array([plain[s][p_col] for s in steps])
    hs = np.array([plain[s][hazard_col] for s in steps])
    peak = int(np.argmax(ps))
    rising_p, rising_h = np.maximum.accumulate(ps[: peak + 1]), hs[: peak + 1]
    return float(np.interp(p_ge1, rising_p, rising_h))


def main() -> None:
    plain = series("none_inloop.tsv")
    print("run step group p_ge1 hazard plain_hazard_at_same_p_ge1 ratio")
    for run in sys.argv[1:] or ["ulwide10_inloop.tsv", "ulwide100_inloop.tsv"]:
        rows = series(run)
        for step in (10, 15, 20, 25, 30):
            for group, (hazard_col, p_col) in (("tool", (0, 1)), ("swe", (2, 3))):
                hazard, p_ge1 = rows[step][hazard_col], rows[step][p_col]
                ref = matched(plain, group, p_ge1)
                print(f"{run.split('_')[0]} {step} {group} {p_ge1:.3f} {hazard:.3f} {ref:.3f} {hazard / ref:.2f}")


if __name__ == "__main__":
    main()
