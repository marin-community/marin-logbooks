# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import time

from experiments.datakit.decontam.encoder import key_set_survey as ks
from experiments.datakit.decontam.encoder.dense_survey import dense_index

L = "work/locators.parquet"
shard = "work/speed/normalized/x/outputs/main/part-0.parquet"
out = "work/e6local"
t = time.time()
print(ks.count_shard(dense_index(L), shard, out), round(time.time() - t, 1), "s count")
t = time.time()
r = ks.scan_shard(L, out + "/counts", shard, out)
print({k: v for k, v in r.items() if k != "path"}, round(time.time() - t, 1), "s scan")
