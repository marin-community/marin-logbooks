# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F13: set 29, the judged removal check of the records that only the proposed key set removes (fresh shards,
mean iteration-16 score >= 0.6, not removed by v11). As set 27: per record its best add-on pair; up to 35 records
per eval family drawn uniformly, plus 200 drawn uniformly; the same rubric and judges. Writes ``work/set29``
(pairs.parquet, batches) and ``strata.json`` (catalog-weighted add-on removals per family).
"""
import io
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from experiments.datakit.decontam.encoder.labels import write_batches
from experiments.datakit.decontam.pair_text import item_texts

T = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore"
C = "s3://marin-us-east-08a/tmp/ttl=7d/rav/decon-poc/catalog-v11-full"
D = "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc"
PER_STRATUM, UNIFORM, BATCHES = 35, 200, 4


def source(path: str) -> str:
    return path.split("/normalized/", 1)[1].split("/outputs/", 1)[0]


removed = pd.read_parquet("work/fresh_addon_removed.parquet")
old = pq.read_table(
    io.BytesIO(StoragePath(f"{D}/decon-poc/catalog-v11-full/removed-final-06.parquet").read_bytes()),
    columns=["path", "record"],
).to_pandas()
old = set(zip(old["path"], old["record"].astype(str)))
removed = removed[[(p, str(r)) not in old for p, r in zip(removed["path"], removed["record"])]]
best = removed.sort_values("score", ascending=False).drop_duplicates(["path", "record"]).copy()
best["family"] = best["eval"].str.split(r"[_ -]").str[0]
best["stratum"] = best["family"] + ":high"
shards_per_source = Counter(source(p) for p in json.loads(StoragePath(f"{C}/shards.json").read_text()))
best["weight"] = best["path"].map(lambda p: shards_per_source[source(p)])
counts = best.groupby("stratum")["weight"].sum().to_dict()
per_stratum = best.sample(frac=1, random_state=29).groupby("stratum").head(PER_STRATUM)
uniform = best.sample(min(UNIFORM, len(best)), random_state=290)
chosen = pd.concat([per_stratum.assign(sample="stratum"), uniform.assign(sample="uniform")])
chosen = chosen.drop_duplicates(["path", "record", "item"]).reset_index(drop=True)
texts = item_texts(f"{D}/decon-g1v7/items")
for field in ("question", "choices", "answer", "passage"):
    chosen[field] = [getattr(texts[i], field) for i in chosen["item"]]
chosen["uid"] = [f"s29-{n:05d}" for n in range(len(chosen))]
directory = Path("work/set29")
write_batches(chosen.sample(frac=1, random_state=291), directory, BATCHES)
(directory / "strata.json").write_text(json.dumps(counts, indent=1))
print(
    f"add-on-only removals: {len(best)} records in {best['family'].nunique()} families; chosen {len(chosen)} ({int((chosen['sample'] == 'stratum').sum())} by stratum, {int((chosen['sample'] == 'uniform').sum())} uniform)"
)
