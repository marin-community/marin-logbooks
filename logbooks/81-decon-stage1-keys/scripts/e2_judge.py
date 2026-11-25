# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import io
import json
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from marin.datakit.decon_keys import text_stream, window_hashes
from rigging.filesystem.storage_path import StoragePath

loc = pq.read_table("work/locators.parquet", columns=["item", "kind", "question"]).to_pandas()
qs = loc[loc["kind"] == "question"].drop_duplicates("item").set_index("item")["question"]
P = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore/encoding-survey-2"
rows = []
for p in StoragePath(f"{P}/new/*.parquet").glob():
    rows += pq.read_table(io.BytesIO(p.read_bytes())).to_pylist()


def share(item, right):
    q = qs.get(item, "")
    w = set(window_hashes(q).tolist())
    if not w:
        return 0.0
    return len(w & set(window_hashes(text_stream(right)).tolist())) / len(w)


by = defaultdict(list)
for r in rows:
    r["share"] = share(r["item"], r["right"])
    by[r["decoding"]].append(r)
for d, rs in by.items():
    s = np.array([r["share"] for r in rs])
    print(
        f"{d:16s} n={len(rs):4d}  share>=0.5: {(s >= 0.5).sum():3d}  share>=0.8: {(s >= 0.8).sum():3d}  median {np.median(s):.2f}"
    )
json.dump(rows, open("work/e2_new_scored.json", "w"))
for r in sorted(rows, key=lambda r: -r["share"])[:14]:
    print("=" * 80)
    print(r["decoding"], r["item"], f"share {r['share']:.2f}", r["path"].split("/normalized/")[-1][:55])
    print("Q:", qs.get(r["item"], "")[:160])
    print("X:", r["right"][:420].replace("\n", " | "))
