# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from rigging.filesystem.storage_path import StoragePath

D = "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc/decon-poc/planted-v11-edits"
paths = list(StoragePath(f"{D}/stats/*.json").glob())
with ThreadPoolExecutor(32) as pool:
    stats = list(pool.map(lambda p: json.loads(p.read_text()), paths))
counts = Counter()
for s in stats:
    counts.update(s["counts"])
items = sum(s["items"] for s in stats)
forms = sorted({k.split("|")[0] for k in counts if "|" in k})
print(f"{len(stats)} evals, {items} sampled items, not indexable {counts['not_indexable']}")
layout = [f for f in forms if not f.startswith("edited/")]
planted = sum(counts[f"{f}|planted"] for f in layout)
print(
    f"layout forms ({len(layout)}): key1 {sum(counts[f'{f}|found'] for f in layout) / planted:.4f}, key1|key2 {sum(counts[f'{f}|found_either'] for f in layout) / planted:.4f}, planted {planted}"
)
for f in forms:
    n = counts[f"{f}|planted"]
    print(f"  {f:32s} {n:6d}  key1 {counts[f'{f}|found'] / n:.3f}  key1|key2 {counts[f'{f}|found_either'] / n:.3f}")
