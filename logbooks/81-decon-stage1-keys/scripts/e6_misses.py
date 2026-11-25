# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import io
import os

import pandas as pd
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

m = pd.read_parquet(os.environ.get("KS_OUT", "work/keyset_rows.parquet"))
T = os.environ.get("KS", "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore/keyset-scan-1")
d = pd.read_parquet("work/dense_scored.parquet")
conf = d[d["score"] >= 0.6].copy()
done = {p.name.removesuffix(".json") for p in StoragePath(f"{T}/stats/*.json").glob()}
conf["name"] = conf["path"].str.split("/normalized/").str[-1].str.replace("/", "__")
conf = conf[conf["name"].isin(done)]
k = pq.read_table(
    io.BytesIO(
        StoragePath(
            "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc/decon-poc/stage1-v11/keys.parquet"
        ).read_bytes()
    )
).to_pandas()
k = k[k["key1_hash"] != 0]
rare_other = set(k[(k["kind"] != "question") & (k["key1_count"] == 0)]["id"])
qcount = k[k["kind"] == "question"].set_index("id")["key1_count"]
j = conf.merge(m[["record", "item", "held", "key_set"]], on=["record", "item"], how="left")
j["held"] = j["held"].fillna(0)
print("stage-2-confirmed key-1 misses on these shards:", len(j), "pairs,", j["record"].nunique(), "records")
m["rare_other"] = m["item"].isin(rare_other)
m["qcount"] = m["item"].map(qcount).fillna(-1)
key1 = m[m["key1"]]
drop_q = (key1["qcount"] > 10) & key1["rare_other"]
for kk in (2, 3, 4, 5):
    caught = j["held"] >= j["key_set"].fillna(8).clip(upper=kk)
    add = m[(m["held"] >= m["key_set"].clip(upper=kk)) & ~m["key1"] & ~m["rare_other"]]
    total = (len(key1) - drop_q.sum() + len(add)) / len(key1)
    print(
        f"  >= {kk} of 8: catches {caught.sum()} of {len(j)} ({caught.mean():.0%}); proposal pairs {total:.2f}x today's key 1 (add {len(add)})"
    )
