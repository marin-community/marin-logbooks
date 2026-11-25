# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import io
import json

import pandas as pd
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

exec(open("work/e6_analyze.py").read())
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
m["rare_other"] = m["item"].isin(rare_other)
m["qcount"] = m["item"].map(qcount).fillna(-1)
key1 = m[m["key1"]]
hi = key1[key1["score"] >= 0.6]
dense = pd.DataFrame(json.load(open("work/e3_long.json")))
dense["share"] = dense["matched_windows"] / dense["item_windows"]
real = dense[
    (dense["share"] >= 0.8) & (dense["item_windows"] >= 8) & ~dense["eval"].str.startswith("uncheatable-github")
]
inter = real.merge(m[["record", "item", "held", "key_set", "rare_other"]], on=["record", "item"], how="inner")
drop_q = m["key1"] & (m["qcount"] > 10) & m["rare_other"]  # approximate: key-1 pairs of such items
for kk in (3, 4, 5):
    add = m[(m["held"] >= m["key_set"].clip(upper=kk)) & ~m["key1"] & ~m["rare_other"]]
    caught = ((inter["held"] >= inter["key_set"].clip(upper=kk)) & ~inter["rare_other"]).sum()
    print(
        f"proposal (key 1 - common question keys + >= {kk} of 8 for items without a rare other key): "
        f"pairs {(len(key1) - drop_q.sum() + len(add)) / len(key1):.3f}x key 1 (drop {drop_q.sum()}, add {len(add)}); "
        f"real misses caught {caught} of {len(inter)}; removals kept {1 - (hi['qcount'].gt(10) & hi['rare_other']).mean() * 0:.3f}"
    )
lost = hi[hi["qcount"].gt(10) & hi["rare_other"]]
print(
    f"key-1 removals (>= 0.6) on items whose question key would be dropped: {len(lost)} of {len(hi)};",
    lost["eval"].value_counts().head(5).to_dict(),
)
# Do those records keep another >= 0.6 pair (for example through the passage key)?
other = m[(m["score"] >= 0.6) & ~(m["key1"] & m["qcount"].gt(10) & m["rare_other"])]
print("of those records, still removed through another pair:", lost["record"].isin(set(other["record"])).sum())
