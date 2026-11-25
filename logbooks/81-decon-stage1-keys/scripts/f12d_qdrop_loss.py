# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F12d: on the fresh shards, the v11 removals (record with a pair at mean score >= 0.6) that the question-key
drop would lose: every pair at >= 0.6 of the record is a dropped question-key pair."""
import io
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

T = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore"
C = "s3://marin-us-east-08a/tmp/ttl=7d/rav/decon-poc/catalog-v11-full"
D = "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc/decon-poc"
read = lambda path, cols=None: pq.read_table(io.BytesIO(StoragePath(path).read_bytes()), columns=cols).to_pandas()
fresh = json.loads(StoragePath(f"{T}/fresh-shards-53.json").read_text())
keys = read(f"{D}/stage1-v11/keys.parquet")
keys = keys[keys["key1_hash"] != 0]
rare_other = set(keys[(keys["kind"] != "question") & (keys["key1_count"] == 0)]["id"])
qcount = keys[keys["kind"] == "question"].set_index("id")["key1_count"]
dropped_items = {i for i in rare_other if qcount.get(i, -1) > 10}


def shard(path: str) -> pd.DataFrame:
    name = path.split("/normalized/", 1)[1].replace("/", "__")
    pairs_path, scores_path = f"{C}/pairs/{name}.parquet", f"{C}/scores-it16/{name}.parquet"
    if not (StoragePath(pairs_path).exists() and StoragePath(scores_path).exists()):
        return pd.DataFrame()
    pairs = read(pairs_path, ["record", "item", "kind"])
    scores = read(scores_path, ["record", "item", "score"])
    return pairs.merge(scores, on=["record", "item"]).assign(path=path)


with ThreadPoolExecutor(32) as pool:
    frame = pd.concat(pool.map(shard, fresh), ignore_index=True)
frame["dropped"] = (frame["kind"] == "question") & frame["item"].isin(dropped_items)
high = frame[frame["score"] >= 0.6]
by_record = high.groupby(["path", "record"])["dropped"].all()
print(
    f"{len(frame):,} v11 pairs with scores; {len(by_record):,} records with a pair >= 0.6; lost by the drop: {int(by_record.sum())}"
)
print(
    "lost records by eval:",
    high[high.set_index(["path", "record"]).index.isin(by_record[by_record].index)]["eval" if "eval" in high else "item"]
    .value_counts()
    .head(8)
    .to_dict(),
)
