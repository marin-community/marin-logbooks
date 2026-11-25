# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import io
import json
import random
from collections import Counter

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

P = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore/dense-survey-1"
tot = Counter()
for p in StoragePath(f"{P}/stats/*.json").glob():
    tot.update({k: v for k, v in json.loads(p.read_text()).items() if isinstance(v, int)})
print(
    {
        k: tot[k]
        for k in (
            "records",
            "key_pairs",
            "dense_pairs",
            "both",
            "dense_only",
            "key_only",
            "key_records",
            "dense_records",
        )
    }
)
rows = []
for p in StoragePath(f"{P}/dense_only/*.parquet").glob():
    rows += pq.read_table(io.BytesIO(p.read_bytes())).to_pylist()
long_rows = [r for r in rows if r["item_windows"] >= 4]
print(len(rows), "dense-only rows;", len(long_rows), "with >= 4 item windows")
print(Counter(r["eval"] for r in long_rows).most_common(20))
loc = pq.read_table("work/locators.parquet", columns=["item", "kind", "question"]).to_pandas()
qs = loc[loc["kind"] == "question"].drop_duplicates("item").set_index("item")["question"]
random.Random(0).shuffle(long_rows)
json.dump(long_rows, open("work/e3_long.json", "w"))
for r in long_rows[:12]:
    print("=" * 90)
    print(
        r["item"],
        f"{r['matched_windows']}/{r['item_windows']}",
        "reading",
        r["reading"],
        r["path"].split("/normalized/")[-1][:60],
    )
    print("Q:", qs.get(r["item"], "")[:240])
    print("X:", r["excerpt"][:600].replace("\n", " | "))
