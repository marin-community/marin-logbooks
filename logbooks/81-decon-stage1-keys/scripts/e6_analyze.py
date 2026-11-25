# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""Key-set (k-of-8) load and recall against key 1, joined to the v11 catalog scores of the same shards."""

import io
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

T = os.environ.get("KS", "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore/keyset-scan-1")
SC = "s3://marin-us-east-08a/tmp/ttl=7d/rav/decon-poc/catalog-v11-full/scores-it16"
tot = Counter()
stats = [json.loads(p.read_text()) for p in StoragePath(f"{T}/stats/*.json").glob()]
for s in stats:
    tot.update({k: v for k, v in s.items() if isinstance(v, int)})
r = tot["records"]
print(len(stats), "shards,", r, "records")
for name in ("key1", "k1", "k2", "k3"):
    print(
        f"  {name}: {tot[name + '_records'] / r:.4%} of records, {tot[name + '_pairs']} pairs ({tot[name + '_pairs'] / tot['key1_pairs']:.2f}x key 1)"
    )
print(
    f"  k2 and key1: {tot['k2_and_key1']} pairs; k2 only: {tot['k2_pairs'] - tot['k2_and_key1']}; key1 only: {tot['key1_pairs'] - tot['k2_and_key1']}"
)
files = [str(p) for p in StoragePath(f"{T}/rows/*.parquet").glob()]


def rows(path):
    return pq.read_table(io.BytesIO(StoragePath(path).read_bytes())).to_pandas()


def scores(name):
    p = StoragePath(f"{SC}/{name}")
    return (
        pq.read_table(io.BytesIO(p.read_bytes()), columns=["record", "item", "score"]).to_pandas()
        if p.exists()
        else None
    )


with ThreadPoolExecutor(32) as pool:
    df = pd.concat(list(pool.map(rows, files)), ignore_index=True)
    sc = [s for s in pool.map(scores, [f.rsplit("/", 1)[1] for f in files]) if s is not None]
sc = pd.concat(sc, ignore_index=True).groupby(["record", "item"])["score"].max().reset_index()
m = df.merge(sc, on=["record", "item"], how="left")
k1 = m[m["key1"]]
print("key-1 pairs with a v11 score:", k1["score"].notna().sum(), "of", len(k1))
hi = k1[k1["score"] >= 0.6]
print(
    f"key-1 pairs >= 0.6: {len(hi)}; held >= 2 of the key set: {(hi['held'] >= hi['key_set'].clip(upper=2)).mean():.3f}; held == key set: {(hi['held'] >= hi['key_set']).mean():.3f}"
)
lost = hi[hi["held"] < hi["key_set"].clip(upper=2)]
print("removals k2 would lose (by eval):", Counter(lost["eval"]).most_common(10))
only = m[(~m["key1"]) & (m["held"] >= m["key_set"].clip(upper=2))]
print(
    f"k2-only pairs: {len(only)}; full key set held: {(only['held'] >= only['key_set']).sum()}; by eval:",
    Counter(only["eval"]).most_common(12),
)
m.to_parquet(os.environ.get("KS_OUT", "work/keyset_rows.parquet"))
