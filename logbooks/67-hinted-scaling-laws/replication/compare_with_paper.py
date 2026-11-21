"""Compare our hinted GSM8K Delphi rollouts with the paper's released per-problem results, cell by cell.

Usage: python compare_with_paper.py <paper results/rollouts dir> <our results dir> [out.csv]

For each rung and (method, H) cell: our accuracy, the paper's accuracy for the same checkpoint (seed
"original", or "0" for 5e19 and 6e20, whose released repos are seed 0), a paired z-score over the 256
problems, and the per-problem correlation of pass rates.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_FILES = {"none": "gsm8k__no_hint.csv", "truncation": "gsm8k__truncate.csv", "masking": "gsm8k__mask.csv"}
SEED_ZERO_RUNGS = {"delphi_5e19", "delphi_6e20"}


def pass_rate(passed: pd.Series) -> pd.Series:
    return passed.map(lambda bits: bits.count("1") / len(bits))


def paper_table(rollouts: Path) -> pd.DataFrame:
    frames = []
    for method, filename in PAPER_FILES.items():
        frame = pd.read_csv(rollouts / filename, dtype={"passed": str, "seed": str})
        if method == "none":
            frame["hint"] = 0.0
        frame["method"] = method
        wanted = np.where(frame["model"].isin(SEED_ZERO_RUNGS), "0", "original")
        frame = frame[frame["seed"] == wanted]
        frames.append(frame[["model", "method", "hint", "problem_id", "passed"]])
    paper = pd.concat(frames)
    paper["paper_rate"] = pass_rate(paper["passed"])
    return paper.drop(columns="passed")


def our_table(results: Path) -> pd.DataFrame:
    paths = [path for path in sorted(results.glob("delphi_*.csv")) if path.stat().st_size > 0]
    ours = pd.concat(pd.read_csv(path, dtype={"passed": str}) for path in paths)
    ours["our_rate"] = pass_rate(ours["passed"])
    return ours.drop(columns="passed")


def compare(paper: pd.DataFrame, ours: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "method", "hint", "problem_id"]
    ours["hint"], paper["hint"] = ours["hint"].round(1), paper["hint"].round(1)
    merged = ours.merge(paper, on=keys, how="left", validate="one_to_one")
    if merged["paper_rate"].isna().any():
        raise ValueError(f"{merged['paper_rate'].isna().sum()} of our cells have no paper counterpart")
    rows = []
    for (model, method, hint), cell in merged.groupby(["model", "method", "hint"]):
        diff = cell["our_rate"] - cell["paper_rate"]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        both_vary = cell["our_rate"].std() > 0 and cell["paper_rate"].std() > 0
        rows.append(
            {
                "model": model,
                "method": method,
                "hint": hint,
                "problems": len(cell),
                "ours": cell["our_rate"].mean(),
                "paper": cell["paper_rate"].mean(),
                "diff": diff.mean(),
                "z": diff.mean() / se if se > 0 else 0.0,
                "corr": np.corrcoef(cell["our_rate"], cell["paper_rate"])[0, 1] if both_vary else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    table = compare(paper_table(Path(sys.argv[1])), our_table(Path(sys.argv[2])))
    if len(sys.argv) > 3:
        table.to_csv(sys.argv[3], index=False)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    flagged = table[table["z"].abs() > 3]
    print(f"\n{len(table)} cells; {len(flagged)} with |z| > 3; median corr {table['corr'].median():.3f}")


if __name__ == "__main__":
    main()
