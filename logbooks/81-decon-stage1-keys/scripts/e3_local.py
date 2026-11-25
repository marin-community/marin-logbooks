# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import resource
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.datakit.decontam.encoder.dense_survey import dense_index, make_dense

S1 = Path("work")
start = time.time()
index = dense_index(str(S1 / "locators.parquet"))
print(
    "index",
    round(time.time() - start),
    "s;",
    len(index.holders),
    "windows;",
    int((index.windows_of > 0).sum()),
    "items; peak MB",
    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 2**20,
)
items = pq.read_table(next(iter(sorted((S1 / "items").glob("arc*")))), columns=["id", "question"]).to_pylist()[:3]
host = "The committee met on Tuesday to review the budget. " * 30
q = items[0]["question"]
words = q.split()
edited = " ".join(words[: len(words) // 3]) + " " + "XYZ " + " ".join(words[len(words) // 3 :])
rows = [{"id": "plain", "text": host}, {"id": "copy", "text": host + q}, {"id": "edited", "text": host + edited}]
root = S1 / "e3local/normalized/src/outputs/main"
root.mkdir(parents=True, exist_ok=True)
pq.write_table(pa.Table.from_pylist(rows), root / "part-00000.parquet")


class Shard:
    shard_idx = 0


start = time.time()
out = list(
    make_dense(str(S1 / "locators.parquet"), str(S1 / "e3local/out"))(iter([str(root / "part-00000.parquet")]), Shard())
)
print(round(time.time() - start, 1), "s", {k: v for k, v in out[0].items() if k != "path"})
