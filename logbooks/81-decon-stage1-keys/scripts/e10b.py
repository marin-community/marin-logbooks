# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""E10: a TF-IDF / BM25-style stage 1 (inverted index of each item's corpus-rare words) on real data."""

import glob
import io
import math
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from marin.datakit.decon_keys import KeyTable, record_readings
from rigging.filesystem.storage_path import StoragePath

S1 = "work"
WORD = re.compile(r"[a-z0-9]{2,}")


def words(text):
    return set(WORD.findall(text.lower()))


t0 = time.time()
records = []
for f in sorted(glob.glob(f"{S1}/corpus/*.parquet")):
    records += [
        r for r in pq.read_table(f, columns=["id", "text"]).to_pylist() if r["text"] and len(r["text"]) < 2_000_000
    ]
N = len(records)
df = Counter()
rec_words = []
for r in records:
    w = words(r["text"])
    rec_words.append(w)
    df.update(w)
print(N, "records;", len(df), "distinct words;", round(time.time() - t0), "s")
items = []
for f in glob.glob(f"{S1}/allitems/*.parquet"):
    items += pq.read_table(f, columns=["id", "eval", "question"]).to_pylist()
idf = lambda w: math.log((N + 1) / (df.get(w, 0) + 1))
import os

DMAX = int(os.environ.get("DMAX", "5"))
postings = defaultdict(list)
item_weight = {}
for it in items:
    ws = words(it["question"] or "")
    item_weight[it["id"]] = sum(idf(w) for w in ws) or 1.0
    for w in ws:
        if df.get(w, 0) <= DMAX:
            postings[w].append(it["id"])
postings = {w: ids for w, ids in postings.items() if len(ids) <= 50}
print(len(items), "items;", len(postings), "indexed rare words;", round(time.time() - t0), "s")


def candidates(ws):
    shared = defaultdict(float)
    count = Counter()
    for w in ws:
        for it in postings.get(w, ()):
            shared[it] += idf(w)
            count[it] += 1
    return shared, count


RULES = [(2, 0.0), (3, 0.0), (5, 0.0)]
load = Counter()
for ws in rec_words:
    shared, count = candidates(ws)
    for m, tau in RULES:
        load[(m, tau)] += sum(1 for it, s in shared.items() if count[it] >= m and s / item_weight[it] >= tau)
# Key-1 load on the same records.
loc = pq.read_table(f"{S1}/locators.parquet")
table = KeyTable(loc.column("key1_hash").to_numpy(), np.arange(loc.num_rows))
key_items = np.array(loc.column("item").to_pylist(), dtype=object)
key1 = 0
for r in records[:: max(1, N // 20000)]:
    hashes = np.concatenate([x.hashes for x in record_readings(r["text"])])
    key1 += len({key_items[i] for i in table.items_of(table.found(hashes))})
key1_scale = key1 * (N / len(records[:: max(1, N // 20000)]))
print(f"key-1 pairs on these records (estimated from a subsample): {key1_scale:,.0f}")
for m, tau in RULES:
    print(
        f"  rare-word rule (>= {m} shared rare words, >= {tau:.0%} of the item's idf): {load[(m, tau)]:,} pairs ({load[(m, tau)] / max(key1_scale, 1):.1f}x key 1)"
    )
# Recall on the stage-2-confirmed key-1 misses and on current removals.
q = {it["id"]: it["question"] or "" for it in items}


def fires(item, text, m, tau):
    ws = words(text) & words(q.get(item, ""))
    rare = [w for w in ws if df.get(w, 0) <= DMAX]
    return len(rare) >= m and sum(idf(w) for w in rare) / item_weight.get(item, 1.0) >= tau


misses = pd.read_parquet(f"{S1}/dense_scored.parquet")
misses = misses[misses["score"] >= 0.6]
B = "s3://marin-us-east-08a/tmp/ttl=7d/rav/decon-poc/catalog-v11-full"
names = sorted(p.name for p in StoragePath(f"{B}/scores-it16/*.parquet").glob())
sample = random.Random(9).sample(names, 400)


def joined(name):
    s_ = pq.read_table(
        io.BytesIO(StoragePath(f"{B}/scores-it16/{name}").read_bytes()), columns=["record", "item", "score"]
    ).to_pandas()
    s_ = s_[s_["score"] >= 0.6]
    if s_.empty:
        return s_.assign(right=pd.Series(dtype=str))
    pname = name.removesuffix(".parquet")
    pth = StoragePath(f"{B}/pairs/{pname}")
    if not pth.exists():
        pth = StoragePath(f"{B}/pairs/{name}")
    p_ = pq.read_table(io.BytesIO(pth.read_bytes()), columns=["record", "item", "right"]).to_pandas()
    return s_.merge(p_, on=["record", "item"], how="inner")


with ThreadPoolExecutor(32) as pool:
    sc = pd.concat(list(pool.map(joined, sample)), ignore_index=True)
removals = sc[sc["score"] >= 0.6]
print(len(misses), "confirmed key-1 misses;", len(removals), "current removal pairs sampled")
for m, tau in RULES:
    rm = np.mean([fires(r.item, r.right, m, tau) for r in misses.itertuples()])
    rr = np.mean([fires(r.item, r.right, m, tau) for r in removals.itertuples()])
    print(f"  (>= {m}, >= {tau:.0%}): recall on key-1 misses {rm:.2f}, on current removals {rr:.2f}")
print(round(time.time() - t0), "s total")
