# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import time
import urllib.parse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.datakit.decontam.encoder.encoding_survey import make_survey

S1 = Path("work")
items = pq.read_table(S1 / "items/", columns=["id", "question"]).to_pylist()[:2]
host = "The committee met on Tuesday to review the budget. " * 30
rows = [
    {"id": "plain", "text": host},
    {"id": "url", "text": host + " https://x.org/?q=" + urllib.parse.quote(items[0]["question"]) + "&a=1&b=2"},
    {"id": "tags", "text": host + " ".join(f"<b>{w}</b>" for w in items[1]["question"].split())},
]
root = S1 / "e2local/normalized/src/outputs/main"
root.mkdir(parents=True, exist_ok=True)
pq.write_table(pa.Table.from_pylist(rows), root / "part-00000.parquet")


class Shard:
    shard_idx = 0


start = time.time()
out = list(
    make_survey(str(S1 / "locators.parquet"), str(S1 / "e2local/out"))(iter([str(root / "part-00000.parquet")]), Shard())
)
print(round(time.time() - start, 1), "s")
print({k: v for k, v in out[0].items() if k != "samples"})
print(pq.read_table(next((S1 / "e2local/out/new").glob("*.parquet"))).select(["record", "item", "decoding"]).to_pylist())
