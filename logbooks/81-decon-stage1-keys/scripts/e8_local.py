# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import numpy as np

from experiments.datakit.decontam.encoder import key_set_survey as ks

L = "work/locators.parquet"
out = "work/e6local"
index, total, keep = ks.key_sets(L, out + "/counts", "spread")
sizes = np.bincount(index.holders[keep], minlength=len(index.items))
print("spread key sets: rows", int(keep.sum()), "size distribution", np.bincount(np.minimum(sizes, 9))[:10].tolist())
n = index.items.index("asdiv-validation-250")
rows = np.nonzero((index.holders == n) & keep)[0]
print(
    "asdiv-validation-250 key-set positions:",
    sorted(index.positions[rows].tolist()),
    "of",
    int(index.windows_of[n]),
    "windows",
)
r = ks.scan_shard(L, out + "/counts", "work/speed/normalized/x/outputs/main/part-0.parquet", "work/e8local", "spread")
print({k: v for k, v in r.items() if k != "path"})
