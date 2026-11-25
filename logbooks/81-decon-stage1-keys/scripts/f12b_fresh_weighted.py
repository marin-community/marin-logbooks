# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F12b: exact question-key drop on the fresh shards (from the v11 catalog pair files, which keep the matched
stream), and catalog-weighted counts: each fresh shard stands for its source's shards."""
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

T = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore"
C = "s3://marin-us-east-08a/tmp/ttl=7d/rav/decon-poc/catalog-v11-full"
D = "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc/decon-poc"


def read(path: str, columns=None) -> pd.DataFrame:
    return pq.read_table(io.BytesIO(StoragePath(path).read_bytes()), columns=columns).to_pandas()


def source(path: str) -> str:
    return path.split("/normalized/", 1)[1].split("/outputs/", 1)[0]


fresh = json.loads(StoragePath(f"{T}/fresh-shards-53.json").read_text())
shards_per_source = Counter(source(p) for p in json.loads(StoragePath(f"{C}/shards.json").read_text()))
weight = {p: shards_per_source[source(p)] for p in fresh}
keys = read(f"{D}/stage1-v11/keys.parquet")
keys = keys[keys["key1_hash"] != 0]
rare_other = set(keys[(keys["kind"] != "question") & (keys["key1_count"] == 0)]["id"])
qcount = keys[keys["kind"] == "question"].set_index("id")["key1_count"]
dropped_items = {i for i in rare_other if qcount.get(i, -1) > 10}


def pairs_of(path: str) -> pd.DataFrame:
    name = path.split("/normalized/", 1)[1].replace("/", "__")
    target = f"{C}/pairs/{name}.parquet"
    frame = (
        read(target, ["record", "item", "kind"])
        if StoragePath(target).exists()
        else pd.DataFrame(columns=["record", "item", "kind"])
    )
    return frame.assign(path=path)


with ThreadPoolExecutor(32) as pool:
    pairs = pd.concat(pool.map(pairs_of, fresh), ignore_index=True)
pairs["dropped"] = (pairs["kind"] == "question") & pairs["item"].isin(dropped_items)
pairs["w"] = pairs["path"].map(weight)
addon = read(f"{T}/fresh-pairs-1/pairs/addon.parquet", ["path", "record", "item"]).assign(
    w=lambda f: f["path"].map(weight)
)
print(
    f"v11 pairs on the fresh shards: {len(pairs):,}; question-key drop {int(pairs['dropped'].sum()):,} ({pairs['dropped'].mean():.1%}); add-on {len(addon):,}"
)
print(f"proposed / v11 pairs, unweighted: {(len(pairs) - pairs['dropped'].sum() + len(addon)) / len(pairs):.2f}x")
weighted = (pairs["w"].sum() - pairs.loc[pairs["dropped"], "w"].sum() + addon["w"].sum()) / pairs["w"].sum()
print(f"proposed / v11 pairs, catalog-weighted: {weighted:.2f}x")
removed = read(f"{D}/catalog-v11-full/removed-final-06.parquet", ["path", "record"])
here = removed[removed["path"].isin(set(fresh))].drop_duplicates()
new = pd.read_parquet("work/fresh_addon_removed.parquet")[["path", "record"]].drop_duplicates()
new = new.merge(here.assign(old=True), on=["path", "record"], how="left")
new = new[new["old"].isna()]
print(f"v11 removals at 0.6 on these shards: {len(here):,}; records only the add-on removes: {len(new):,}")
w_old = here["path"].map(weight).sum()
w_new = new["path"].map(weight).sum()
print(f"catalog-weighted: v11 removals {w_old:,.0f}, add-on only {w_new:,.0f} (+{w_new / w_old:.1%})")
by_source = new.assign(source=new["path"].map(source)).groupby("source").size().sort_values(ascending=False)
print("add-on-only removals by source:", by_source.head(8).to_dict())
