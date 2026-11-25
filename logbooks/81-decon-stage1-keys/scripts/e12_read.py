# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import glob

import pandas as pd
import pyarrow.parquet as pq

S1 = "work"
m = pd.read_parquet(f"{S1}/dense_scored.parquet")
m = m[m["score"] >= 0.6]
ids = set(m["item"])
q = {}
for f in glob.glob(f"{S1}/allitems/*.parquet"):
    for r in pq.read_table(f, columns=["id", "question"]).to_pylist():
        if r["id"] in ids:
            q[r["id"]] = r["question"] or ""
print(m.columns.tolist(), len(m))
for r in m.sample(16, random_state=5).itertuples():
    text = getattr(r, "right", None) or getattr(r, "excerpt", "")
    print("=" * 100)
    print(f"[{r.eval}] score {r.score:.2f}")
    print("ITEM:  ", " ".join(q.get(r.item, "")[:260].split()))
    print("RECORD:", " ".join(str(text)[:420].split()))
