# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import io

import pandas as pd
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

k = pq.read_table(
    io.BytesIO(
        StoragePath(
            "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc/decon-poc/stage1-v11/keys.parquet"
        ).read_bytes()
    )
).to_pandas()
k = k[k["key1_hash"] != 0]
q = k[k["kind"] == "question"].set_index("id")["key1_count"]
other = k[(k["kind"] != "question") & (k["key1_count"] == 0)].groupby("id").size()
per_item = pd.read_parquet("work/per_item_pairs.parquet")["pairs"]
items = (
    pd.DataFrame({"pairs": per_item}).join(q.rename("qcount")).join(other.rename("rare_other")).fillna({"rare_other": 0})
)
total = items["pairs"].sum()
print("items in the sample:", len(items), "pairs:", total)
all_q = q
for t in (10, 100, 1000):
    heavy = items["qcount"] > t
    with_other = heavy & (items["rare_other"] > 0)
    print(
        f"question key count > {t:>4}: {heavy.sum():>6} items in sample, {items.loc[heavy, 'pairs'].sum() / total:.1%} of pairs;"
        f" of which with a rare passage/choices key: {items.loc[with_other, 'pairs'].sum() / total:.1%};"
        f" catalog-wide items with such keys: {(all_q > t).sum()}"
    )
