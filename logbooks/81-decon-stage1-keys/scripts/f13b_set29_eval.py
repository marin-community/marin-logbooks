# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F13b: precision of set 29 (removals only the proposed key set makes, fresh shards), as set 27: a record counts
when both judges agree; strata are eval families weighted by catalog-weighted removals, with an item-clustered
bootstrap 5% bound; the uniform sample as a check; and the split by source."""
import json

import numpy as np
import pandas as pd
from rigging.filesystem.storage_path import StoragePath

L = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore/labels/set29"
GOOD, BAD = {"COPY", "EDITED"}, {"VARIANT", "GENERIC", "UNRELATED"}


def labels(judge: str) -> pd.DataFrame:
    rows = [
        json.loads(l)
        for p in StoragePath(f"{L}/{judge}/batch_*.jsonl").glob()
        for l in p.read_text().split("\n")
        if l.strip()
    ]
    return pd.DataFrame(rows).set_index("uid")[["label", "note"]].add_prefix(judge.split("-")[0] + "_")


pairs = pd.read_parquet("work/set29/pairs.parquet").set_index("uid")
frame = pairs.join(labels("glm-5.3"), how="inner").join(labels("deepseek-v4.1-flash"), how="inner")
good = frame.glm_label.isin(GOOD) & frame.deepseek_label.isin(GOOD)
bad = frame.glm_label.isin(BAD) & frame.deepseek_label.isin(BAD)
print(f"judged {len(frame)}: copy {int(good.sum())}, not {int(bad.sum())}, split {int((~good & ~bad).sum())}")
labels_of = frame.assign(both=frame.glm_label + "/" + frame.deepseek_label)
frame = frame[good | bad].assign(y=good[good | bad].astype(int)).reset_index()
counts = pd.Series(json.loads(open("work/set29/strata.json").read()))
strat = frame[frame["sample"] == "stratum"]
rng = np.random.default_rng(0)


def weighted(part: pd.DataFrame) -> tuple[float, float]:
    by = part.groupby("stratum")
    weights = counts.reindex(list(by.groups)).fillna(0)
    weights = weights / weights.sum()
    point = float(sum(weights[s] * g["y"].mean() for s, g in by))
    items = {s: g.groupby("item")["y"].apply(list).tolist() for s, g in by}
    boot = []
    for _ in range(2000):
        total = 0.0
        for s, groups in items.items():
            picked = [v for k in rng.integers(0, len(groups), len(groups)) for v in groups[k]]
            total += weights[s] * (sum(picked) / len(picked))
        boot.append(total)
    return point, float(np.percentile(boot, 5))


p, low = weighted(strat)
print(
    f"add-on-only removals at 0.6: precision {p:.3f} [5% bound {low:.3f}]: {'PASS' if p >= 0.75 and low >= 0.65 else 'FAIL'} (floor 0.75, bound 0.65)"
)
uni = frame[frame["sample"] == "uniform"]
print(f"uniform sample: {len(uni)} labeled, precision {uni.y.mean():.3f}")
frame["source"] = frame["path"].str.split("/normalized/").str[1].str.split("/outputs/").str[0]
print(
    "by source:",
    frame.groupby("source")["y"]
    .agg(["size", "mean"])
    .sort_values("size", ascending=False)
    .head(6)
    .round(2)
    .to_dict("index"),
)
by = strat.groupby("stratum")["y"].agg(["size", "mean"])
by["weighted_removals"] = counts.reindex(by.index).fillna(0).astype(int)
print(by.sort_values("weighted_removals", ascending=False).head(16).round(2).to_string())
print("judge labels:", labels_of["both"].value_counts().head(8).to_dict())

# Record-weighted estimate: each judged stratum record stands for (its source's catalog weight) x (records of its
# stratum / judged records of its stratum); the uniform weighting above under-weights sources with many shards.
best = pd.read_parquet("work/set29/pairs.parquet")
population = pd.read_parquet("work/fresh_addon_removed.parquet").drop_duplicates(["path", "record"])
stratum_size = population.assign(family=population["eval"].str.split(r"[_ -]").str[0]).groupby("family").size()
strat = strat.assign(family=strat["stratum"].str.removesuffix(":high"))
judged = strat.groupby("family").size()
strat["ht"] = strat["weight"] * strat["family"].map(stratum_size) / strat["family"].map(judged)
print(
    f"record-weighted precision: {(strat['ht'] * strat['y']).sum() / strat['ht'].sum():.3f} over {len(strat)} judged records"
)
by_eval = strat.groupby("eval").agg(judged=("y", "size"), precision=("y", "mean"), weight=("ht", "sum"))
print(by_eval.sort_values("weight", ascending=False).head(14).round(2).to_string())
low = {"qnli", "wnli", "squad", "Humanity's", "uncheatable"}
keep = ~strat["family"].isin(low)
print(
    f"without {sorted(low)}: record-weighted precision {(strat.loc[keep, 'ht'] * strat.loc[keep, 'y']).sum() / strat.loc[keep, 'ht'].sum():.3f}; weight kept {strat.loc[keep, 'ht'].sum() / strat['ht'].sum():.2f}"
)
