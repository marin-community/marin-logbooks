# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F6b: for removals whose item has an [asy] drawing, the share of the item's words outside the drawing that the
excerpt holds; a low share means the excerpt shares only the drawing code."""
import glob
import io
import re

import pandas as pd
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from experiments.datakit.decontam.encoder.labels import word_overlap

T = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore"
ASY = re.compile(r"\[asy\].*?\[/asy\]", re.DOTALL)
pairs = pq.read_table(io.BytesIO(StoragePath(f"{T}/asy-removals-1/pairs.parquet").read_bytes())).to_pandas()
items = pd.concat(
    [pq.read_table(f, columns=["id", "question"]).to_pandas() for f in glob.glob("work/allitems/*.parquet")]
)
question = dict(zip(items["id"], items["question"]))
pairs["text_share"] = [word_overlap(ASY.sub(" ", question[i]), r or "") for i, r in zip(pairs["item"], pairs["right"])]
pairs["boilerplate"] = pairs["right"].fillna("").str.contains("rr_cartesian_axes|real ticklen=3", regex=True)
bins = pd.cut(pairs["text_share"], [-0.01, 0.5, 0.8, 0.95, 1.0], labels=["<0.5", "0.5-0.8", "0.8-0.95", ">=0.95"])
print(len(pairs), "removal pairs;", int(pairs["right"].isna().sum()), "without an excerpt")
print(pd.crosstab(bins, pairs["boilerplate"], margins=True).to_string())
print(pairs[pairs["text_share"] < 0.5].groupby("eval").size().sort_values(ascending=False).head(8).to_dict())
pairs.to_parquet("work/asy_pairs.parquet")
