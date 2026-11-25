# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F5e: the judged copies of set 30 (stage 1 never sent them): their spread over shards, a shard-clustered
bootstrap of the catalog estimate, and whether key 1 or key 2 or the proposed key set holds them."""
import io
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from marin.datakit.decon_keys import KeyTable, record_readings
from rigging.filesystem.storage_path import StoragePath

sys.path.insert(0, "logbooks/81-decon-stage1-keys/scripts")
from f8_regression import _proposed  # noqa: E402

D = "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc/decon-poc"
T = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore"
frame = pd.read_parquet("work/set30_judged.parquet")
labeled = frame[frame["copy"] | frame["clean"]]
copies = labeled[labeled["copy"]]
print(
    f"judged copies: {len(copies)} in {copies['shard'].nunique()} shards and {copies['shard'].str.split('/normalized/').str[1].str.split('/outputs/').str[0].nunique()} sources"
)
print(
    copies.groupby(copies["shard"].str.split("/normalized/").str[1].str.split("/outputs/").str[0])
    .size()
    .sort_values(ascending=False)
    .head(8)
    .to_dict()
)
rng = np.random.default_rng(0)
for stratum in ("model", "sim_high"):
    part = labeled[labeled["stratum"] == stratum]
    shards = part.groupby("shard")
    per_shard = [(g["weight"] * g["copy"]).sum() / len(part) for _, g in shards]
    # Estimate = (population weight) x (copy share); bootstrap the share over shards.
    population_weight = pd.read_json(
        io.StringIO(
            StoragePath("s3://marin-us-east-02a/tmp/ttl=7d/rav/stage1-explore/embed-2/set30/population.json").read_text()
        )
    ).at[stratum, "weighted"]
    groups = [g["copy"].to_numpy() for _, g in shards]
    boot = []
    for _ in range(2000):
        pick = [groups[k] for k in rng.integers(0, len(groups), len(groups))]
        values = np.concatenate(pick)
        boot.append(values.mean() * population_weight)
    print(
        f"{stratum}: estimate {part['copy'].mean() * population_weight:,.0f}; shard bootstrap 5-95% {np.percentile(boot, 5):,.0f} - {np.percentile(boot, 95):,.0f} ({part['shard'].nunique()} shards)"
    )
keys = pq.read_table(io.BytesIO(StoragePath(f"{D}/stage1-v11/keys.parquet").read_bytes())).to_pandas()
k1 = keys[keys["key1_hash"] != 0]
k2 = keys[(keys["key2_hash"] != 0) & (keys["key2_count"] == 0)]
t1 = KeyTable(k1["key1_hash"].to_numpy(np.uint64), k1["id"].to_numpy(object))
t2 = KeyTable(k2["key2_hash"].to_numpy(np.uint64), k2["id"].to_numpy(object))
proposed = _proposed("work/locators.parquet", f"{T}/keyset-count-1/counts", 100, keys)
hits = []
for text, item in zip(copies["right"], copies["item"]):
    h = np.concatenate([r.hashes for r in record_readings(text)])
    one = item in set(t1.items_of(t1.found(h)))
    hits.append((one, one or item in set(t2.items_of(t2.found(h))), proposed(item, h)))
hits = np.array(hits)
print(
    f"of {len(copies)} judged copies, the excerpt holds: key 1 {hits[:, 0].sum()}, key 1 or key 2 {hits[:, 1].sum()}, the proposed key set {hits[:, 2].sum()}"
)
print(copies.assign(proposed=hits[:, 2]).groupby("stratum")["proposed"].agg(["size", "sum"]).to_dict("index"))
