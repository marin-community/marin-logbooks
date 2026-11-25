# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F1: one fresh shard per source (seed 53) that no earlier step used: the key reference shards (2 per source,
seed 11), the measure shards (seed 7), and every shard of the stage-1 surveys of 2026-11-24."""
import json
import random

from marin.datakit.decon import reference_shards
from marin.datakit.decon_tables import parquet_files
from rigging.filesystem.storage_path import StoragePath, prefix_join

ROOT = "s3://marin-us-east-02a/marin/normalized"
T = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore"
names = json.load(open("experiments/datakit/decontam/encoder/sources.json"))
dirs = [prefix_join(ROOT, f"{name}/outputs/main") for name in names]
used = set(reference_shards(dirs, 2, 11))
for listing in ("dense-survey-1/shards.json",):
    used |= set(json.loads(StoragePath(f"{T}/{listing}").read_text()))
# Shards that set or tuned the keys and key sets; a small source falls back to shards outside them only.
tuning = set(used)
for listing in ("encoding-survey-2/shards.json", "bpe-survey-1/shards.json"):
    path = StoragePath(f"{T}/{listing}")
    if path.exists():
        used |= set(json.loads(path.read_text()))
for directory in ("keyset-count-1/counts", "keyset-spread-1/stats"):
    for path in StoragePath(f"{T}/{directory}/*").glob():
        shard = prefix_join(ROOT, path.name.split(".parquet")[0].replace("__", "/") + ".parquet")
        used.add(shard)
        tuning.add(shard)
rng = random.Random(53)
fresh, measure_rng, fallbacks = [], random.Random(7), 0
for directory in dirs:
    shards = parquet_files(directory, "**/*.parquet")
    measure = set(measure_rng.sample(shards, 1))
    pool = [s for s in shards if s not in used and s not in measure]
    fallback = [s for s in shards if s not in tuning and s not in measure]
    fresh.append(rng.choice(pool) if pool else rng.choice(fallback) if fallback else None)
    fallbacks += not pool and bool(fallback)
fresh = [s for s in fresh if s]
print(len(used), "used shards known;", len(fresh), "fresh shards;", fallbacks, "from encoding-survey shards")
StoragePath(f"{T}/fresh-shards-53.json").write_text(json.dumps(fresh))
