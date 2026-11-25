# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F11: eval items whose question stream is mostly CJK, their stream lengths and key status (review point 4)."""
import glob
import io
import re

import pandas as pd
import pyarrow.parquet as pq
from marin.datakit.decon_keys import text_stream
from rigging.filesystem.storage_path import StoragePath

CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
items = pd.concat(
    [
        pq.read_table(f, columns=["id", "eval", "question", "choices"]).to_pandas()
        for f in glob.glob("work/allitems/*.parquet")
    ]
)
items["stream"] = [text_stream(q or "") for q in items["question"]]
items["cjk_share"] = [len(CJK.findall(s)) / max(len(s), 1) for s in items["stream"]]
cjk = items[items["cjk_share"] >= 0.5].copy()
keys = pq.read_table(
    io.BytesIO(
        StoragePath(
            "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc/decon-poc/stage1-v11/keys.parquet"
        ).read_bytes()
    )
).to_pandas()
keyed = set(keys[keys["key1_hash"] != 0]["id"])
cjk["keyed"] = cjk["id"].isin(keyed)
cjk["chars"] = cjk["stream"].str.len()
print(f"{len(cjk):,} CJK items in {cjk['eval'].nunique()} evals; keyed {cjk['keyed'].mean():.3f}")
print(
    cjk.groupby("eval")
    .agg(items=("id", "size"), keyed=("keyed", "mean"), median_chars=("chars", "median"))
    .sort_values("items", ascending=False)
    .head(15)
    .round(2)
    .to_string()
)
cjk[["id", "eval", "chars", "keyed"]].to_parquet("work/cjk_items.parquet")
