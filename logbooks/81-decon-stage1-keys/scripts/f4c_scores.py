# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F4c: stage-2 scores (mean of the two iteration-16 models) of the planted paraphrases and translations."""
import io

import pandas as pd
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

T = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore/paraphrase-1"
read = lambda name: pq.read_table(io.BytesIO(StoragePath(f"{T}/{name}").read_bytes())).to_pandas()
s0, s1 = read("scored_s0.parquet"), read("scored_s1.parquet")
frame = s0.assign(score=(s0["score_label_q"] + s1["score_label_q"]) / 2)
frame["flag"] = frame["score"] >= 0.6
frame["both"] = frame["flag"] & frame["proposed"]
summary = frame.groupby("group").agg(
    rows=("flag", "size"), stage2_at_06=("flag", "mean"), median_score=("score", "median"), stage1_and_2=("both", "mean")
)
print(summary.round(3).to_string())
print(frame[frame.group == "translation"].groupby("kind")["flag"].mean().round(3).to_string())
frame.to_parquet("work/paraphrase_scored.parquet")
