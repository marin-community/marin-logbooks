# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F16: the proposed key set on the fresh shards with counts that skip likely copies (keyset-fresh-cap100-nocopy)
against the counts with copies (keyset-fresh-cap100): add-on pairs and the confirmed key-1 misses caught."""
import io
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

T = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore"


def read_all(prefix: str) -> pd.DataFrame:
    paths = list(StoragePath(f"{prefix}/*.parquet").glob())
    with ThreadPoolExecutor(32) as pool:
        return pd.concat(
            pool.map(
                lambda p: pq.read_table(io.BytesIO(p.read_bytes()), columns=["record", "item", "eval"]).to_pandas(),
                paths,
            ),
            ignore_index=True,
        )


misses = pd.read_parquet("work/fresh_misses.parquet")
for label, run in (("with copies", "keyset-fresh-cap100"), ("without copies", "keyset-fresh-cap100-nocopy")):
    addon = read_all(f"{T}/{run}/pairs")
    admitted = set(zip(addon["record"], addon["item"]))
    caught = sum((r, i) in admitted for r, i in zip(misses["record"], misses["item"]))
    print(
        f"{label}: add-on pairs {len(addon):,} in {addon['record'].nunique():,} records; misses caught {caught} of {len(misses)}"
    )
a = read_all(f"{T}/keyset-fresh-cap100/pairs")
b = read_all(f"{T}/keyset-fresh-cap100-nocopy/pairs")
only_b = b.merge(a, on=["record", "item", "eval"], how="left", indicator=True).query("_merge == 'left_only'")
only_a = a.merge(b, on=["record", "item", "eval"], how="left", indicator=True).query("_merge == 'left_only'")
print(f"pairs only without copies: {len(only_b):,} {only_b['eval'].value_counts().head(6).to_dict()}")
print(f"pairs only with copies: {len(only_a):,} {only_a['eval'].value_counts().head(6).to_dict()}")
