# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import io

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from marin.datakit.decon_keys import KeyTable, record_readings
from rigging.filesystem.storage_path import StoragePath

S1 = "work"
k = pq.read_table(
    io.BytesIO(
        StoragePath(
            "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc/decon-poc/stage1-v11/keys.parquet"
        ).read_bytes()
    )
).to_pandas()
k1 = k[k["key1_hash"] != 0]
k2 = k[(k["key2_hash"] != 0) & (k["key2_count"] == 0)]
t1 = KeyTable(k1.key1_hash.to_numpy(np.uint64), k1.id.to_numpy(object))
t2 = KeyTable(k2.key2_hash.to_numpy(np.uint64), k2.id.to_numpy(object))
m = pd.read_parquet(f"{S1}/dense_scored.parquet")
m = m[m["score"] >= 0.6]
m = m[~m["right"].str.contains("rr_cartesian_axes|real ticklen=3", regex=True)]
a = b = 0
by_eval = {}
for r in m.itertuples():
    h = np.concatenate([x.hashes for x in record_readings(r.right)])
    x1 = r.item in set(t1.items_of(t1.found(h)))
    x2 = r.item in set(t2.items_of(t2.found(h)))
    a += x1
    b += x1 or x2
    e = by_eval.setdefault(r.eval, [0, 0])
    e[0] += 1
    e[1] += x1 or x2
print(
    f"{len(m)} confirmed non-boilerplate misses: key1 fires on the excerpt {a}, key1|key2(count 0) {b} ({b / len(m):.0%})"
)
print({e: f"{v[1]}/{v[0]}" for e, v in sorted(by_eval.items(), key=lambda kv: -kv[1][0])[:8]})
