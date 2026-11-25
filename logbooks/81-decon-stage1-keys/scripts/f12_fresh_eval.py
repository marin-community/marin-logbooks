# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F12: the proposed stage 1 on the fresh shards (seed 53): load against key 1, the stage-2-confirmed key-1
misses it catches, and the records it newly removes (mean iteration-16 score >= 0.6)."""
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

T = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore"
K = "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc/decon-poc/stage1-v11/keys.parquet"
BOILERPLATE = re.compile(r"rr_cartesian_axes|real ticklen=3")
CUTOFF = 0.6


def read(path: str) -> pd.DataFrame:
    return pq.read_table(io.BytesIO(StoragePath(path).read_bytes())).to_pandas()


def read_all(prefix: str) -> pd.DataFrame:
    paths = list(StoragePath(f"{prefix}/*").glob())
    with ThreadPoolExecutor(32) as pool:
        return pd.concat(pool.map(lambda p: read(str(p)), paths), ignore_index=True)


s0, s1 = read(f"{T}/fresh-pairs-1/scored_s0.parquet"), read(f"{T}/fresh-pairs-1/scored_s1.parquet")
scored = s0.assign(score=(s0["score_label_q"] + s1["score_label_q"]) / 2)
scored["boilerplate"] = scored["right"].fillna("").str.contains(BOILERPLATE)
stats = [json.loads(p.read_text()) for p in StoragePath(f"{T}/keyset-fresh-cap100/stats/*.json").glob()]
records = sum(s.get("records", 0) for s in stats)
key1_pairs = sum(s.get("key1_pairs", 0) for s in stats)
addon_pairs = sum(s.get("addon_pairs", 0) for s in stats)
rows = read_all(f"{T}/keyset-fresh-cap100/rows")
keys = read(K)
keys = keys[keys["key1_hash"] != 0]
rare_other = set(keys[(keys["kind"] != "question") & (keys["key1_count"] == 0)]["id"])
qcount = keys[keys["kind"] == "question"].set_index("id")["key1_count"]
key1_rows = rows[rows["key1"]]
dropped = int(((key1_rows["item"].map(qcount).fillna(-1) > 10) & key1_rows["item"].isin(rare_other)).sum())
print(
    f"{len(stats)} shards, {records:,} records; key 1: {key1_pairs:,} pairs; question-key drop: -{dropped:,}; add-on: +{addon_pairs:,}"
)
print(f"proposed / key 1 pairs: {(key1_pairs - dropped + addon_pairs) / key1_pairs:.2f}x")
dense = scored[scored["source"] == "dense"]
misses = dense[(dense["score"] >= CUTOFF) & ~dense["boilerplate"]]
addon = scored[scored["source"] == "addon"]
admitted = set(zip(addon["record"], addon["item"]))
misses = misses.assign(caught=[(r, i) in admitted for r, i in zip(misses["record"], misses["item"])])
print(
    f"confirmed key-1 misses (dense-only, score >= {CUTOFF}, no [asy] boilerplate): {len(misses)} pairs in {misses['record'].nunique()} records; caught {int(misses['caught'].sum())}"
)
print(
    "misses by eval:",
    misses.groupby("eval")["caught"].agg(["size", "sum"]).sort_values("size", ascending=False).head(10).to_dict("index"),
)
removed = addon[addon["score"] >= CUTOFF]
print(
    f"add-on pairs at >= {CUTOFF}: {len(removed)} in {removed['record'].nunique()} records (of {addon['record'].nunique():,} add-on records); [asy] boilerplate {int(removed['boilerplate'].sum())}"
)
removed = removed.assign(family=removed["eval"].str.split(r"[_ -]").str[0])
print(
    "new removals by family:",
    removed.groupby("family")["record"].nunique().sort_values(ascending=False).head(15).to_dict(),
)
removed.to_parquet("work/fresh_addon_removed.parquet")
misses.to_parquet("work/fresh_misses.parquet")
