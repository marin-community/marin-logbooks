# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F4b: plant the LLM paraphrases and translations of F4 (and the plain item as a control) into real record
text, check stage 1 (key 1; key 1 or key 2 at count 0; the proposed key set), and write the planted pairs for
stage-2 scoring (``score.py --pairs <output>``)."""
import io
import random
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from marin.datakit.decon_keys import KeyTable, record_readings
from marin.datakit.decon_tables import parquet_files, read_frame, write_frame
from rigging.filesystem.storage_path import StoragePath

from experiments.datakit.decontam.encoder.planted import CONTEXT, MIN_HOST, lettered

sys.path.insert(0, "logbooks/81-decon-stage1-keys/scripts")
from f8_regression import _proposed  # noqa: E402

T = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore"
D = "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc"
LOCATORS = f"{D}/decon-poc/catalog-v11-full/locators.parquet"
keys = pq.read_table(io.BytesIO(StoragePath(f"{D}/decon-poc/stage1-v11/keys.parquet").read_bytes())).to_pandas()
rewrites = read_frame(f"{T}/paraphrase-1/rewrites.parquet")
items = pd.concat([read_frame(p, ["id", "question", "choices"]) for p in parquet_files(f"{D}/decon-g1v7/items")])
items = items.set_index("id")
hosts = [
    t for t in read_frame(f"{D}/decon-poc/data/planted_hosts.parquet", ["text"])["text"] if t and len(t) >= MIN_HOST
]
k1 = keys[keys["key1_hash"] != 0]
k2 = keys[(keys["key2_hash"] != 0) & (keys["key2_count"] == 0)]
t1 = KeyTable(k1["key1_hash"].to_numpy(np.uint64), k1["id"].to_numpy(object))
t2 = KeyTable(k2["key2_hash"].to_numpy(np.uint64), k2["id"].to_numpy(object))
locators = StoragePath(LOCATORS)
local = "work/locators.parquet"
proposed = _proposed(local, f"{T}/keyset-count-1/counts", 100, keys)
plain = rewrites.drop_duplicates("item").assign(kind="plain")
plain["question"] = [items.at[i, "question"] for i in plain["item"]]
plain["choices"] = [list(items.at[i, "choices"]) if items.at[i, "choices"] is not None else [] for i in plain["item"]]
rng = random.Random(9)
rows = []
for row in pd.concat([plain, rewrites], ignore_index=True).itertuples():
    host = rng.choice(hosts)
    text = lettered(row.question, list(row.choices))
    record = f"{host[:CONTEXT]}\n{text}\n{host[CONTEXT:2 * CONTEXT]}"
    hashes = np.concatenate([r.hashes for r in record_readings(record)])
    one = row.item in set(t1.items_of(t1.found(hashes)))
    rows.append(
        {
            "record": f"planted-{len(rows)}",
            "item": row.item,
            "eval": row.eval,
            "kind": row.kind,
            "right": record,
            "key1": one,
            "key1_or_key2": one or row.item in set(t2.items_of(t2.found(hashes))),
            "proposed": proposed(row.item, hashes),
        }
    )
planted = pd.DataFrame(rows)
planted["group"] = planted["kind"].str.split(":").str[0]
write_frame(planted, f"{T}/paraphrase-1/pairs/pairs.parquet")
print(
    planted.groupby("group")[["key1", "key1_or_key2", "proposed"]]
    .mean()
    .round(3)
    .assign(rows=planted.groupby("group").size())
    .to_string()
)
print(planted[planted.group == "translation"].groupby("kind")[["key1", "proposed"]].mean().round(3).to_string())
