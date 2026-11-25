# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import io
import random
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

P = "s3://marin-us-east-08a/tmp/ttl=7d/rav/decon-poc/catalog-v11-full/scores-it16"
files = sorted(str(p) for p in StoragePath(f"{P}/*.parquet").glob())
sample = random.Random(5).sample(files, 2000)


def read(path):
    return pq.read_table(io.BytesIO(StoragePath(path).read_bytes()), columns=["record", "item", "score"]).to_pandas()


with ThreadPoolExecutor(32) as pool:
    df = pd.concat(list(pool.map(read, sample)), ignore_index=True)
k = pq.read_table(
    io.BytesIO(
        StoragePath(
            "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc/decon-poc/stage1-v11/keys.parquet"
        ).read_bytes()
    )
).to_pandas()
k = k[k["key1_hash"] != 0]
q = k[k["kind"] == "question"].set_index("id")["key1_count"]
rare_other = set(k[(k["kind"] != "question") & (k["key1_count"] == 0)]["id"])
df["qcount"] = df["item"].map(q).fillna(-1)
df["has_rare_other"] = df["item"].isin(rare_other)
hi = df["score"] >= 0.6
best = df.groupby("record")["score"].max()
removed = set(best[best >= 0.6].index)
for t in (10, 100, 1000):
    drop = (df["qcount"] > t) & df["has_rare_other"]
    # Records that would lose their only >= 0.6 pair if these pairs were dropped.
    kept_hi = df[hi & ~drop].groupby("record").size()
    lost = {r for r in df.loc[hi & drop, "record"] if r not in kept_hi.index}
    print(
        f"count > {t:>4}: drops {drop.mean():.1%} of pairs, {(hi & drop).sum()} of {hi.sum()} pairs >= 0.6,"
        f" {len(lost)} of {len(removed)} removed records lose their only >= 0.6 pair"
    )
df[(df["qcount"] > 10) & df["has_rare_other"] & hi].to_parquet("work/heavy_hi.parquet")
