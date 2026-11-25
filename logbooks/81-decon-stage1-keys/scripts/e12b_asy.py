# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import pandas as pd

m = pd.read_parquet("work/dense_scored.parquet")
m = m[m["score"] >= 0.6]
asy = m["right"].str.contains("rr_cartesian_axes|real ticklen=3", regex=True)
print(f"confirmed misses: {len(m)} pairs, {m.record.nunique()} records")
print(f"asy-boilerplate pairs: {int(asy.sum())} ({asy.mean():.0%}); evals:", m[asy]["eval"].value_counts().to_dict())
print("non-asy by eval:", m[~asy]["eval"].value_counts().head(8).to_dict())
print(f"non-asy records: {m[~asy].record.nunique()}")
