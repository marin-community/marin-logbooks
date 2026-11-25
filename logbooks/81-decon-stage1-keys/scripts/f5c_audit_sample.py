# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F5c: set 30, the judge sample of the embedding audit. One pair per record (its highest-similarity pair that is
not a stage-1 candidate, among the pairs with an excerpt), the calibration shards left out, in strata:
  - ``model``: mean iteration-16 score >= 0.6 (copies stage 2 would remove but stage 1 never sent);
  - ``sim_high``: score < 0.6 and similarity >= 0.85 (where paraphrases and translations should be);
  - ``sim_mid``: score < 0.6 and similarity 0.80-0.85;
  - ``random``: the shards' random excerpt draws, any similarity.
Each record carries its catalog weight (shard records / sampled records x catalog shards / sampled shards).
"""
import io
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from pathlib import Path

from experiments.datakit.decontam.encoder.labels import write_batches
from experiments.datakit.decontam.pair_text import item_texts

E = "s3://marin-us-east-02a/tmp/ttl=7d/rav/stage1-explore/embed-2"
C = "s3://marin-us-east-08a/tmp/ttl=7d/rav/decon-poc/catalog-v11-full"
D = "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc"
PER_STRATUM = {"model": 300, "sim_high": 300, "sim_mid": 150, "random": 150}
read = lambda path: pq.read_table(io.BytesIO(StoragePath(path).read_bytes())).to_pandas()
s0, s1 = read(f"{E}/scored_s0.parquet"), read(f"{E}/scored_s1.parquet")
pairs = s0.assign(score=(s0["score_label_q"] + s1["score_label_q"]) / 2)
pairs = pairs[~pairs["shard"].str.contains("/stage1-explore/embed-2/cal/")]
catalog_shards = len(json.loads(StoragePath(f"{C}/shards.json").read_text()))
sampled_shards = pairs["shard"].nunique()
pairs["weight"] = pairs["shard_records"] / pairs["sampled_records"] * catalog_shards / 400
pairs["stratum"] = np.select(
    [pairs["selection"] == "random", pairs["score"] >= 0.6, pairs["sim"] >= 0.85, pairs["sim"] >= 0.80],
    ["random", "model", "sim_high", "sim_mid"],
    default="low",
)
best = pairs.sort_values(["score", "sim"], ascending=False).drop_duplicates(["shard", "record", "stratum"])
population = best.groupby("stratum").agg(records=("record", "size"), weighted=("weight", "sum"))
print(f"{len(pairs):,} scored pairs in {sampled_shards} shards; records per stratum:\n{population.round(0).to_string()}")
chosen = pd.concat(
    [
        best[best["stratum"] == s].sample(min(n, int((best["stratum"] == s).sum())), random_state=30)
        for s, n in PER_STRATUM.items()
    ]
).reset_index(drop=True)
texts = item_texts(f"{D}/decon-g1v7/items")
for field in ("question", "choices", "answer", "passage"):
    chosen[field] = [getattr(texts[i], field) for i in chosen["item"]]
chosen["path"] = chosen["shard"]
chosen["uid"] = [f"s30-{n:05d}" for n in range(len(chosen))]
directory = Path("set30")
write_batches(chosen.sample(frac=1, random_state=301), directory, 4)
population.to_json(directory / "population.json")
# Upload the set (pairs, batches, population) next to the audit outputs.
for path in sorted(directory.rglob("*")):
    if path.is_file():
        StoragePath(f"{E}/{path}").write_bytes(path.read_bytes())
print(f"chosen {len(chosen)}: {chosen['stratum'].value_counts().to_dict()}")
