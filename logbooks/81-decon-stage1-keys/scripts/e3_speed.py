# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import io
import json
import time

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from experiments.datakit.decontam.encoder import dense_survey as d

shards = json.loads(
    StoragePath("s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore/dense-survey-1/shards.json").read_text()
)
path = shards[0]
t = pq.read_table(io.BytesIO(StoragePath(path).read_bytes()), columns=["id", "text"])
print(path.split("/normalized/")[-1], t.num_rows, "records")
rows = t.slice(0, 2000).to_pylist()
local = "work/speed/normalized/x/outputs/main"
import os

os.makedirs(local, exist_ok=True)
import pyarrow as pa

pq.write_table(pa.Table.from_pylist(rows), f"{local}/part-0.parquet")
d.dense_index("work/locators.parquet")


class S:
    shard_idx = 0


start = time.time()
out = list(d.make_dense("work/locators.parquet", "work/speed/out")(iter([f"{local}/part-0.parquet"]), S()))
dt = time.time() - start
print(
    f"{dt:.1f} s for 2,000 records -> {t.num_rows * dt / 2000 / 60:.0f} min per shard",
    {k: v for k, v in out[0].items() if k != "path"},
)
