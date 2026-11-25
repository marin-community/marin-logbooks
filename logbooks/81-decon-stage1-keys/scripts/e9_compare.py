# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import io
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

B = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore"
runs = {
    "spread": f"{B}/keyset-spread-1",
    "spread+cap10": f"{B}/keyset-spreadcap-1",
    "spread+cap100": f"{B}/keyset-spreadcap100-1",
}
names = None
for T in runs.values():
    got = {p.name.removesuffix(".json") for p in StoragePath(f"{T}/stats/*.json").glob()}
    names = got if names is None else names & got
print(len(names), "common shards")
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
d = pd.read_parquet("work/dense_scored.parquet")
d["name"] = d["path"].str.split("/normalized/").str[-1].str.replace("/", "__")
conf = d[(d["score"] >= 0.6) & d["name"].isin(names)]
print("confirmed misses on these shards:", len(conf), "pairs")
for label, T in runs.items():

    def read(n):
        p = StoragePath(f"{T}/rows/{n}.parquet")
        return pq.read_table(io.BytesIO(p.read_bytes())).to_pandas() if p.exists() else None

    with ThreadPoolExecutor(32) as pool:
        m = pd.concat([x for x in pool.map(read, sorted(names)) if x is not None], ignore_index=True)
    m["rare_other"] = m["item"].isin(rare_other)
    m["qcount"] = m["item"].map(qcount).fillna(-1)
    key1 = m[m["key1"]]
    drop_q = ((key1["qcount"] > 10) & key1["rare_other"]).sum()
    j = conf.merge(m[["record", "item", "held", "key_set"]], on=["record", "item"], how="left")
    j["held"] = j["held"].fillna(0)
    for kk in (3, 4, 5):
        need = m["key_set"].clip(upper=kk)
        add = m[(m["held"] >= need) & (m["key_set"] >= 1) & ~m["key1"] & ~m["rare_other"]]
        caught = (j["held"] >= j["key_set"].fillna(8).clip(upper=kk)) & (j["key_set"].fillna(8) >= 1)
        print(
            f"{label:13s} >= {kk} of 8: pairs {(len(key1) - drop_q + len(add)) / len(key1):.2f}x today's key 1; misses caught {caught.sum()} of {len(j)}; top add evals {add['eval'].value_counts().head(3).to_dict()}"
        )
