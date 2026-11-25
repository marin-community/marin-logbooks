# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F14: spread key sets (cap 100) from the counts of keyset-count-1 and from the recount that skips likely
copies (keyset-count-nocopy, a record holding >= 3 of an item's windows does not count for that item): items
with no key set or a short one, and the shift in key-set windows (review point 3)."""
import numpy as np
import pandas as pd

from experiments.datakit.decontam.encoder import key_set_survey as ks

T = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore"
LOCATORS = "work/locators.parquet"
results = {}
for label, counts in (
    ("with copies", f"{T}/keyset-count-1/counts"),
    ("without copies", f"{T}/keyset-count-nocopy/counts"),
):
    ks.key_sets.cache_clear()
    index, total, keep = ks.key_sets(LOCATORS, counts, "spread", 100)
    size = np.bincount(index.holders[keep], minlength=len(index.items))
    has = np.bincount(index.holders, minlength=len(index.items)) > 0
    slot_of_row = np.searchsorted(index.starts, np.arange(len(index.holders)), side="right") - 1
    frame = pd.DataFrame({"holder": index.holders, "count": total[slot_of_row]})
    median = frame.groupby("holder")["count"].median().reindex(range(len(index.items))).fillna(0).to_numpy()
    results[label] = (size, median, has)
    print(
        f"{label}: items with windows {int(has.sum()):,}; no key set {int(((size == 0) & has).sum()):,}; fewer than 3 windows {int(((size < 3) & has).sum()):,}"
    )
size_a, median_a, has = results["with copies"]
size_b, median_b, _ = results["without copies"]
buckets = pd.cut(median_a, [-0.5, 0.5, 10, 100, 1e12], labels=["0", "1-10", "11-100", ">100"])
table = pd.DataFrame(
    {"bucket": buckets, "empty_with": (size_a == 0) & has, "empty_without": (size_b == 0) & has, "has": has}
)
print(
    table[table["has"]]
    .groupby("bucket", observed=True)[["empty_with", "empty_without"]]
    .agg(["sum", "mean"])
    .round(3)
    .to_string()
)
buckets_b = pd.cut(median_b, [-0.5, 0.5, 10, 100, 1e12], labels=["0", "1-10", "11-100", ">100"])
print("median window count bucket, with copies (rows) against without copies (columns):")
print(pd.crosstab(buckets[has], buckets_b[has]).to_string())
