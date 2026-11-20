"""Reproduce the Hinted Scaling Laws paper's fits with marin.scaling_laws.hinted_fit.

Usage: python -I reproduce_paper.py <hinted-scaling-laws checkout>
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from marin.scaling_laws.hinted_fit import (
    ForecastKind,
    HintMethod,
    LadderScores,
    ScoreCells,
    bootstrap_forecasts,
    fit_joint_law,
    fit_ladder,
)

ROOT = Path(sys.argv[1])
RESULTS = ROOT / "data" / "results"
FIT_ROLLOUTS = {"aime": 10, "hle": 10, "gsm8k": None}
METHOD_FILE = {HintMethod.MASKING: "mask", HintMethod.TRUNCATION: "truncate"}
TABLE_PAIRS = [("gsm8k", "delphi"), ("aime", "qwen25"), ("aime", "gemma3"), ("hle", "qwen3"), ("hle", "qwen35")]

# Printed by the paper's experiments/forecasting/table1.py at c0490e53:
# pair, method: b1 b2 b0 A L F, joint MAE (pp)
PAPER_TABLE1 = {
    ("gsm8k/delphi", "mask"): (0.89, 4.20, -3.95, 1.000, 0.031, 0.0222, 4.11),
    ("gsm8k/delphi", "truncate"): (1.73, 2.46, -6.53, 1.000, 0.006, 0.0059, 1.05),
    ("aime/qwen25", "mask"): (1.89, 5.28, 32.66, 1.000, 0.018, 0.0195, 3.41),
    ("aime/qwen25", "truncate"): (1.64, 7.26, 25.44, 1.000, 0.046, 0.0000, 5.67),
    ("aime/gemma3", "mask"): (2.15, 3.84, 38.21, 1.000, 0.020, 0.0062, 3.66),
    ("aime/gemma3", "truncate"): (1.87, 3.79, 32.73, 1.000, 0.009, 0.0000, 5.01),
    ("hle/qwen3", "mask"): (1.57, 5.99, 26.78, 0.866, 0.014, 0.0250, 1.69),
    ("hle/qwen3", "truncate"): (1.28, 5.67, 20.55, 0.933, 0.026, 0.0000, 1.73),
    ("hle/qwen35", "mask"): (2.52, 4.91, 45.62, 0.883, 0.012, 0.0295, 2.88),
    ("hle/qwen35", "truncate"): (2.00, 3.46, 36.26, 0.925, -0.043, 0.0362, 2.85),
}
# experiments/forecasting/plot.py forecasts, gsm8k/delphi: t -> (baseline AE, ensemble AE)
PAPER_DELPHI_FORECAST = {7: (0.1931, 0.0735), 8: (0.2081, 0.0355), 9: (0.2184, 0.0171), 10: (0.2210, 0.0606), 11: (0.1165, 0.0189)}
# experiments/forecasting/two_stage.py: t -> two-stage (no reference models) AE
PAPER_TWO_STAGE = {9: 0.1628, 8: 0.2300, 7: 0.2304, 6: 0.2347, 5: 0.2364}
TWO_STAGE_RUNGS = ["3e18", "9e18", "2e19", "3e19", "9e19", "2e20", "3e20", "1e21", "1e22", "1e23"]


def read_rollouts(fp: Path) -> pd.DataFrame:
    w = pd.read_csv(fp, dtype=str, keep_default_na=False)
    n = w["passed"].str.len().to_numpy()
    long = w.loc[w.index.repeat(n), [c for c in w.columns if c not in ("passed", "pool")]].reset_index(drop=True)
    long["sample_id"] = np.concatenate([np.arange(k) for k in n])
    long["passed"] = (np.frombuffer("".join(w["passed"]).encode(), dtype="S1") == b"1").astype(float)
    return long


def problem_table(bench: str, method: str, seed: str | None = None) -> pd.DataFrame:
    """(model, hint) x problem matrix of per-problem accuracy: mean over rollouts, then over seeds."""
    df = read_rollouts(RESULTS / "rollouts" / f"{bench}__{method}.csv")
    if FIT_ROLLOUTS[bench] is not None:
        df = df[df["sample_id"] < FIT_ROLLOUTS[bench]]
    if seed is not None:
        df = df[df["seed"] == seed]
    df["hint"] = 0.0 if method == "no_hint" else df["hint"].astype(float).round(4)
    if "seed" not in df.columns:
        df["seed"] = ""
    per_seed = df.groupby(["model", "hint", "seed", "problem_id"]).passed.mean()
    return per_seed.groupby(["model", "hint", "problem_id"]).mean().unstack("problem_id")


def cells(table: pd.DataFrame, capability: pd.Series) -> ScoreCells:
    table = table[table.index.get_level_values("model").isin(capability.index)]
    models = table.index.get_level_values("model")
    return ScoreCells(
        capability=capability.loc[models].to_numpy(dtype=float),
        hint=table.index.get_level_values("hint").to_numpy(dtype=float),
        problem_scores=table.to_numpy(dtype=float),
    )


def pair_ladder(bench: str, fam: str, seed: str | None = None, models: list[str] | None = None) -> LadderScores:
    meta = pd.read_csv(RESULTS / "models.csv")
    meta = meta[(meta.benchmark == bench) & (meta.family == fam) & meta.in_forecasts]
    if models is not None:
        meta = meta[meta.model.isin(models)]
    capability = meta.set_index("model")["capability"]
    nh = problem_table(bench, "no_hint", seed)
    nh = nh[nh.index.get_level_values("model").isin(capability.index)]
    order = np.argsort(capability.loc[nh.index.get_level_values("model")].to_numpy())
    nh = nh.iloc[order]
    hinted = {}
    for m, f in METHOD_FILE.items():
        t = problem_table(bench, f, seed)[nh.columns]
        hinted[m] = cells(t, capability)
    if bench == "gsm8k":
        bpb_tab = pd.read_csv(RESULTS / "bpb_delphi.csv").set_index("model")["bits_per_byte"]
        # 5e19 and 6e20 have no released bpb; placeholder 1.0 (the two-stage fit is only compared on the ten rungs)
        bpb = bpb_tab.reindex(nh.index.get_level_values("model")).fillna(1.0).to_numpy(dtype=float)
    else:
        bpb = np.ones(len(nh))
    return LadderScores(unhinted=cells(nh, capability), hinted=hinted, bits_per_byte=bpb)


def main() -> None:
    print("== Table 1: joint law parameters and in-sample MAE ==")
    joint_mae = {"mask": [], "truncate": []}
    worst = 0.0
    for bench, fam in TABLE_PAIRS:
        ladder = pair_ladder(bench, fam)
        for m, f in METHOD_FILE.items():
            both_c = np.r_[ladder.hinted[m].capability, ladder.unhinted.capability]
            both_h = np.r_[ladder.hinted[m].hint, ladder.unhinted.hint]
            both_y = np.r_[ladder.hinted[m].accuracy, ladder.unhinted.accuracy]
            p = fit_joint_law(both_c, both_h, both_y)
            # Table 1 MAE: per-model mean absolute residual, then mean over models
            res = np.abs(p.predict(both_c, both_h) - both_y)
            mae = np.mean([res[np.isclose(both_c, c)].mean() for c in np.unique(both_c)])
            joint_mae[f].append(mae)
            ours = (p.capability_slope, p.hint_slope, p.intercept, p.upper, p.lower, p.floor, 100 * mae)
            ref = PAPER_TABLE1[(f"{bench}/{fam}", f)]
            # the paper prints 2-4 decimals; compare at its printed precision
            prec = (0.005, 0.005, 0.005, 0.0005, 0.0005, 0.00005, 0.005)
            ok = all(abs(a - b) <= tol + 1e-9 for a, b, tol in zip(ours, ref, prec, strict=True))
            worst = max(worst, max(abs(a - b) for a, b in zip(ours, ref, strict=True)))
            print(
                f"{bench}/{fam:8s} {f:9s} ours "
                + " ".join(f"{v:7.3f}" for v in ours)
                + f"  paper " + " ".join(f"{v:7.3f}" for v in ref)
                + ("  MATCH" if ok else "  DIFF")
            )
    print(f"Table 1 joint MAE: masking {100 * np.mean(joint_mae['mask']):.1f} pp (paper 3.1), "
          f"truncation {100 * np.mean(joint_mae['truncate']):.1f} pp (paper 3.3); worst abs param diff {worst:.4f}")

    print("\n== GSM8K x Delphi forecasts of 1e23 (absolute error) ==")
    ladder = pair_ladder("gsm8k", "delphi")
    caps = ladder.unhinted.capability
    target, y_target = caps[-1], ladder.unhinted.accuracy[-1]
    for t in range(7, len(caps)):
        fc = fit_ladder(ladder.up_to(caps[t - 1])).forecasts(target)
        base, ens = abs(fc[ForecastKind.SIGMOID_BASELINE] - y_target), abs(fc[ForecastKind.HINTED_ENSEMBLE] - y_target)
        pb, pe = PAPER_DELPHI_FORECAST[t]
        print(f"t={t:2d} baseline {base:.4f} (paper {pb:.4f})  ensemble {ens:.4f} (paper {pe:.4f})  "
              f"masking {abs(fc[ForecastKind.MASKING] - y_target):.4f} truncation {abs(fc[ForecastKind.TRUNCATION] - y_target):.4f}")

    print("\n== Two-stage baseline, original seed, ten rungs ==")
    ts = pair_ladder("gsm8k", "delphi", seed="original", models=[f"delphi_{r}" for r in TWO_STAGE_RUNGS])
    caps = ts.unhinted.capability
    target, y_target = caps[-1], ts.unhinted.accuracy[-1]
    for t, paper in PAPER_TWO_STAGE.items():
        fc = fit_ladder(ts.up_to(caps[t - 1])).forecasts(target)
        print(f"t={t} two-stage AE {abs(fc[ForecastKind.TWO_STAGE_BASELINE] - y_target):.4f} (paper {paper:.4f})  "
              f"ensemble AE {abs(fc[ForecastKind.HINTED_ENSEMBLE] - y_target):.4f}")

    print("\n== Bootstrap over problems: GSM8K x Delphi original seed, ten rungs, t = 9 ==")
    start = time.time()
    boot = bootstrap_forecasts(ts.up_to(caps[8]), target, 100, 0)
    print(f"100 resamples in {time.time() - start:.0f}s; observed 1e23 accuracy {y_target:.4f}")
    for kind in ForecastKind:
        lo, hi = boot.interval(kind, 0.9)
        print(f"{kind:20s} point {boot.point[kind]:.4f}  90% [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
