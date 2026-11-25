# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import io
import json
from collections import Counter

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

P = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore/dense-survey-1"
tot = Counter()
stats = [json.loads(p.read_text()) for p in StoragePath(f"{P}/stats/*.json").glob()]
for s in stats:
    tot.update({k: v for k, v in s.items() if isinstance(v, int)})
print(len(stats), "shards;", dict(tot))
rows = []
for p in StoragePath(f"{P}/dense_only/*.parquet").glob():
    rows += pq.read_table(io.BytesIO(p.read_bytes())).to_pylist()
print(len(rows), "dense-only rows;", Counter(r["eval"] for r in rows).most_common(20))
json.dump(rows, open("work/e3_dense_only.json", "w"))
