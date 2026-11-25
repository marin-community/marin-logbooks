# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F17: planted layouts and edits at width 16 (keys rebuilt with the production settings, F10) against width 20
(v11), on the same 41,151 items; load proxy: reference records holding the chosen keys, summed."""
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

T = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore"
D = "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc/decon-poc"
runs = {
    "20": (f"{D}/planted-v11-edits-g1v7", f"{D}/stage1-v11/keys.parquet"),
    "16": (f"{T}/keys-w16-v11/planted", f"{T}/keys-w16-v11/keys.parquet"),
}
table = {}
for width, (planted, keys_path) in runs.items():
    with ThreadPoolExecutor(32) as pool:
        stats = list(pool.map(lambda p: json.loads(p.read_text()), StoragePath(f"{planted}/stats/*.json").glob()))
    counts = Counter()
    for s in stats:
        counts.update(s["counts"])
    layouts = [k.split("|")[0] for k in counts if k.endswith("|planted") and not k.startswith("edited/")]
    row = {"layouts": sum(counts[f"{f}|found"] for f in layouts) / sum(counts[f"{f}|planted"] for f in layouts)}
    for form in sorted(k.split("|")[0] for k in counts if k.endswith("|planted") and k.startswith("edited/")):
        row[form.removeprefix("edited/")] = counts[f"{form}|found"] / counts[f"{form}|planted"]
    keys = pq.read_table(
        io.BytesIO(StoragePath(keys_path).read_bytes()), columns=["key1_hash", "key1_count"]
    ).to_pandas()
    keys = keys[keys["key1_hash"] != 0]
    row["key_count_sum"] = int(keys["key1_count"].sum())
    row["keyed_streams"] = len(keys)
    table[width] = row
print(pd.DataFrame(table).round(3).to_string())
