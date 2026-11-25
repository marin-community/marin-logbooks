# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F5b: recall of the embedding audit (bge-m3, top 10 per record) on its calibration shards: the dense-survey
key-1 misses (real copies) and the F4 plants (plain, paraphrase, translation) inside real host text."""
import io

import pandas as pd
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

E = "s3://marin-us-east-02a/tmp/ttl=7d/rav/stage1-explore/embed-2"
read = lambda path: pq.read_table(io.BytesIO(StoragePath(path).read_bytes())).to_pandas()
misses = pd.read_parquet("work/dense_scored.parquet")
misses = misses[misses["score"] >= 0.6].reset_index(drop=True)
plants = pd.read_parquet("work/paraphrase_scored.parquet").reset_index(drop=True)
for name, truth in (("misses", misses.assign(group="miss")), ("plants", plants)):
    found = read(f"{E}/candidates/{name}__outputs__main__part-0.parquet.parquet")
    known = {f"r{i}": (item, group) for i, (item, group) in enumerate(zip(truth["item"], truth["group"]))}
    found["true"] = [known[r][0] == i for r, i in zip(found["record"], found["item"])]
    found["rank"] = found.sort_values("sim", ascending=False).groupby("record").cumcount()
    hit = found[found["true"]]
    groups = pd.Series([g for _, g in known.values()]).value_counts()
    for group, total in groups.items():
        records = hit[[known[r][1] == group for r in hit["record"]]]
        print(
            f"{name}/{group}: top-10 {records['record'].nunique()}/{total}, top-3 {records[records['rank'] < 3]['record'].nunique()}, top-1 {records[records['rank'] == 0]['record'].nunique()}; true sim median {records['sim'].median():.3f}"
        )
    print(
        f"{name}: sim of other rows, median {found.loc[~found['true'], 'sim'].median():.3f}, 99th percentile {found.loc[~found['true'], 'sim'].quantile(0.99):.3f}"
    )
