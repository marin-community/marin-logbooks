# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from rigging.filesystem.storage_path import StoragePath

D = "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc/decon-poc/planted-v11-edits"
with ThreadPoolExecutor(32) as pool:
    stats = list(pool.map(lambda p: json.loads(p.read_text()), StoragePath(f"{D}/stats/*.json").glob()))
plain = Counter(m["eval"] for s in stats for m in s["misses"] if m["form"] == "plain")
print(sum(plain.values()), plain.most_common(12))
