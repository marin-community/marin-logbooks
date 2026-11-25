# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F15: a shorter key window for CJK items (review point 4). Planted recall on the 190 CJK evals with keys built
for those evals at widths 10 and 8 (F10 builds) against the production width 20 (planted-v11-edits-g1v7), and the
candidate load proxy: reference records holding each chosen key (key1_count), summed over the CJK items."""
import io
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

T = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore"
D = "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc/decon-poc"
item_files = list(StoragePath(f"{T}/items-cjk/items/*.parquet").glob())
evals = {p.name.removesuffix(".parquet") for p in item_files}
with ThreadPoolExecutor(32) as pool:
    cjk_ids = set().union(
        *pool.map(
            lambda p: set(pq.read_table(io.BytesIO(p.read_bytes()), columns=["id"]).column("id").to_pylist()), item_files
        )
    )
runs = {"20": f"{D}/planted-v11-edits-g1v7", "10": f"{T}/keys-cjk-w10/planted", "8": f"{T}/keys-cjk-w8/planted"}
keys = {
    "20": f"{D}/stage1-v11/keys.parquet",
    "10": f"{T}/keys-cjk-w10/keys.parquet",
    "8": f"{T}/keys-cjk-w8/keys.parquet",
}
rows = []
for width, prefix in runs.items():
    paths = [p for p in StoragePath(f"{prefix}/stats/*.json").glob() if p.name.removesuffix(".json") in evals]
    if not paths:
        continue
    with ThreadPoolExecutor(32) as pool:
        stats = list(pool.map(lambda p: json.loads(p.read_text()), paths))
    counts, zho = Counter(), Counter()
    for s in stats:
        counts.update(s["counts"])
        if s["eval"].startswith("zhoblimp"):
            zho.update(s["counts"])
    forms = sorted({k.split("|")[0] for k in counts if "|planted" in k and not k.startswith("edited/")})
    planted = sum(counts[f"{f}|planted"] for f in forms)
    row = {"width": width, "evals": len(stats), "layouts": sum(counts[f"{f}|found"] for f in forms) / planted}
    for f in ("plain", "numbered", "json", "question_only"):
        row[f] = counts[f"{f}|found"] / max(counts[f"{f}|planted"], 1)
    zp = sum(zho[f"{f}|planted"] for f in forms)
    row["zhoblimp_layouts"] = sum(zho[f"{f}|found"] for f in forms) / zp if zp else float("nan")
    row["not_indexable"] = counts["not_indexable"]
    k = pq.read_table(
        io.BytesIO(StoragePath(keys[width]).read_bytes()), columns=["id", "key1_hash", "key1_count"]
    ).to_pandas()
    k = k[k["id"].isin(cjk_ids) & (k["key1_hash"] != 0)]
    row["keyed_streams"] = len(k)
    row["key_count_sum"] = int(k["key1_count"].sum())
    row["key_count_0"] = float((k["key1_count"] == 0).mean())
    rows.append(row)
print(pd.DataFrame(rows).set_index("width").round(3).T.to_string())
