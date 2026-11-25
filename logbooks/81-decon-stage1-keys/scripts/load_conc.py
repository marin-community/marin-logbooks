# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import io
import random
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

P = "s3://marin-us-east-08a/tmp/ttl=7d/rav/decon-poc/catalog-v11-full/scores-it16"
files = sorted(str(p) for p in StoragePath(f"{P}/*.parquet").glob())
sample = random.Random(5).sample(files, 2000)


def read(path):
    t = pq.read_table(io.BytesIO(StoragePath(path).read_bytes()))
    cols = [c for c in ("item", "score", "eval") if c in t.column_names]
    return t.select(cols).to_pandas()


with ThreadPoolExecutor(32) as pool:
    frames = list(pool.map(read, sample))
import pandas as pd

df = pd.concat(frames, ignore_index=True)
print(len(files), "files;", len(df), "pairs in the sample; columns", df.columns.tolist())
per_item = df.groupby("item").size().sort_values(ascending=False)
total = per_item.sum()
for top in (10, 100, 1000, 10000):
    print(f"top {top:>6} items hold {per_item.head(top).sum() / total:.1%} of pairs")
score_col = "score" if "score" in df else None
hi = df[df[score_col] >= 0.6] if score_col else df.iloc[:0]
print("pairs >= 0.6:", len(hi), f"({len(hi) / len(df):.3%})")
heavy = per_item.head(1000).index
print("share of >= 0.6 pairs on the top-1000 items:", f"{hi['item'].isin(heavy).mean():.1%}" if len(hi) else "n/a")
print("top items:", per_item.head(15).to_dict())
per_item.to_frame("pairs").to_parquet("work/per_item_pairs.parquet")
