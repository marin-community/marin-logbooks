# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F9: the pairs to score from the fresh shards: dense-only pairs that hold >= 60% of an item's >= 8 windows (the
miss finder, as in E3), and the pairs only the key set admits (``keyset-fresh-cap100/pairs``)."""
import io
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
from marin.datakit.decon_tables import write_frame
from rigging.filesystem.storage_path import StoragePath

T = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore"


def read_all(prefix: str) -> pd.DataFrame:
    paths = list(StoragePath(f"{prefix}/*.parquet").glob())
    with ThreadPoolExecutor(32) as pool:
        return pd.concat(
            pool.map(lambda p: pq.read_table(io.BytesIO(p.read_bytes())).to_pandas(), paths), ignore_index=True
        )


dense = read_all(f"{T}/dense-survey-fresh/dense_only")
dense["share"] = dense["matched_windows"] / dense["item_windows"]
dense = dense[(dense["share"] >= 0.6) & (dense["item_windows"] >= 8)]
dense = dense.rename(columns={"excerpt": "right"})[["path", "record", "item", "eval", "right", "share"]]
addon = read_all(f"{T}/keyset-fresh-cap100/pairs")
print(
    f"dense-only pairs to score: {len(dense):,}; add-on pairs: {len(addon):,} in {addon['record'].nunique():,} records"
)
write_frame(dense.assign(source="dense"), f"{T}/fresh-pairs-1/pairs/dense.parquet")
write_frame(addon.assign(source="addon"), f"{T}/fresh-pairs-1/pairs/addon.parquet")
