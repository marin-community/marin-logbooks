# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F2: the shard list behind keyset-count-1 (from its count file names), for a recount on the same shards."""
import json

from rigging.filesystem.storage_path import StoragePath, prefix_join

ROOT = "s3://marin-us-east-02a/marin/normalized"
T = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore"
shards = sorted(
    prefix_join(ROOT, path.name.removesuffix(".npz").removesuffix(".parquet").replace("__", "/") + ".parquet")
    for path in StoragePath(f"{T}/keyset-count-1/counts/*.npz").glob()
)
assert all(StoragePath(s).exists() for s in shards[:5])
StoragePath(f"{T}/count-shards-1.json").write_text(json.dumps(shards))
print(len(shards), shards[0])
