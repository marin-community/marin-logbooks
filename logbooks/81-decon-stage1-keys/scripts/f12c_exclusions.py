# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F12c: the add-on without the families whose added removals the judges reject (set 29: QNLI, WNLI, SQuAD, HLE,
Uncheatable), on the fresh shards: add-on pairs, added removals, and confirmed misses caught."""
import io

import pandas as pd
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

T = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore"
EXCLUDED = ("qnli", "wnli", "squad", "Humanity's", "uncheatable")
family = lambda evals: evals.str.split(r"[_ -]").str[0]
addon = pq.read_table(
    io.BytesIO(StoragePath(f"{T}/fresh-pairs-1/pairs/addon.parquet").read_bytes()), columns=["record", "item", "eval"]
).to_pandas()
removed = pd.read_parquet("work/fresh_addon_removed.parquet")
misses = pd.read_parquet("work/fresh_misses.parquet")
for label, keep in (("all families", lambda f: f == f), ("without " + ", ".join(EXCLUDED), lambda f: ~f.isin(EXCLUDED))):
    a = addon[keep(family(addon["eval"]))]
    r = removed[keep(family(removed["eval"]))]
    m = misses[keep(family(misses["eval"]))]
    print(
        f"{label}: add-on pairs {len(a):,}; added removals {r['record'].nunique():,} records; misses {len(m)}, caught {int(m['caught'].sum())}"
    )
