# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""E7: would a boundary-safe key (question body only, no eval-wide label inside) be held by the missed copies?"""

import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
from marin.datakit.decon_keys import WINDOW, readings, text_stream, window_hashes
from rigging.filesystem.storage_path import StoragePath

S1 = "work"
rows = json.load(open(f"{S1}/e3_long.json"))
miss = [
    r
    for r in rows
    if r["matched_windows"] / r["item_windows"] >= 0.8
    and r["item_windows"] >= 8
    and not r["eval"].startswith("uncheatable-github")
]
loc = pq.read_table(f"{S1}/locators.parquet", columns=["item", "kind", "eval", "question", "key"]).to_pandas()
loc = loc[loc["kind"] == "question"].drop_duplicates("item").set_index("item")
# Window counts from the key-set count pass (every window of the corpus sample that is in the dense index).
import sys

sys.path.insert(0, ".")
from experiments.datakit.decontam.encoder.dense_survey import dense_index

index = dense_index(f"{S1}/locators.parquet")
total = np.zeros(len(index.hashes), dtype=np.int64)
paths = [
    str(p)
    for p in StoragePath("s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore/keyset-count-1/counts/*.npz").glob()
]
with ThreadPoolExecutor(32) as pool:
    for part in pool.map(lambda p: np.load(io.BytesIO(StoragePath(p).read_bytes())), paths):
        np.add.at(total, part["slots"], part["counts"])


def count_of(h):
    s = np.searchsorted(index.hashes, h)
    return int(total[s]) if s < len(index.hashes) and index.hashes[s] == h else -1


# Eval-wide labels: 8-character pieces that occur in at least half of an eval's question streams.
by_eval = defaultdict(list)
for item, row in loc.iterrows():
    by_eval[row["eval"]].append(row["question"])
need = {r["eval"] for r in miss}
labels = {}
for ev in need:
    qs = by_eval[ev]
    c = Counter()
    for q in qs:
        c.update({q[i : i + 8] for i in range(max(0, len(q) - 7))})
    labels[ev] = {g for g, n in c.items() if n >= 0.5 * len(qs) and len(qs) >= 10}
items_dir = (
    "s3://marin-us-east-02a/marin/datakit/decontam/items/aa-index-v4.1.1-v6+atlas-benchmark-references-v1+r1/items"
)
choices_of = {}
for ev in need:
    t = pq.read_table(
        io.BytesIO(StoragePath(f"{items_dir}/{ev}.parquet").read_bytes()), columns=["id", "choices"]
    ).to_pylist()
    choices_of.update({r["id"]: r["choices"] or [] for r in t})
held_now, held_safe = 0, 0
out = Counter()
for r in miss:
    q = loc.loc[r["item"], "question"]
    body_end = len(q)
    for c in choices_of.get(r["item"], []):
        cs = text_stream(c)
        if len(cs) >= 1:
            pos = q.rfind(cs)
            if pos > 0:
                body_end = min(body_end, pos)
    hashes = window_hashes(q)
    blocked = np.zeros(len(q) + 1, dtype=bool)
    for lab in labels.get(r["eval"], ()):
        start = q.find(lab)
        while start >= 0:
            blocked[start : start + len(lab)] = True
            start = q.find(lab, start + 1)
    best, best_count = None, None
    for i in range(0, max(0, min(len(hashes), body_end - WINDOW + 1))):
        w = q[i : i + WINDOW]
        if blocked[i : i + WINDOW].any():
            continue
        n = count_of(int(hashes[i]))
        if n < 0:
            continue
        if best is None or n < best_count:
            best, best_count = w, n
    streams = [text_stream(x) for x in readings(r["excerpt"])]
    safe_held = best is not None and any(best in s for s in streams)
    held_safe += safe_held
    out[(r["eval"], safe_held)] += 1
print(len(miss), "missed copies; a boundary-safe key would be held by", held_safe)
print(sorted(out.items(), key=lambda kv: -kv[1])[:30])
