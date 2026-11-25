# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""E13: numbers for review points 1-3 of PR #11993 (asartran): planted edits under key1 / key1|key2 / proposed,
key2's pair cost, and key-1 recall on near-certain copies split by how often the item's text is in the reference."""

import glob
import io
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from marin.datakit.decon_keys import MAX_DOC_CHARS, KeyTable, record_readings
from rigging.filesystem.storage_path import StoragePath

sys.path.insert(0, ".")
from experiments.datakit.decontam.encoder import key_set_survey as ks

S1 = "work"
B = "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore"
T0 = time.time()
index, total, keep = ks.key_sets(f"{S1}/locators.parquet", f"{B}/keyset-count-1/counts", "spread", 100)
slot_of_row = np.searchsorted(index.starts, np.arange(len(index.holders)), side="right") - 1
number = {item: n for n, item in enumerate(index.items)}
print("key sets", round(time.time() - T0), "s", flush=True)
k = pq.read_table(
    io.BytesIO(
        StoragePath(
            "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc/decon-poc/stage1-v11/keys.parquet"
        ).read_bytes()
    )
).to_pandas()
k1 = k[k["key1_hash"] != 0]
k2 = k[k["key2_hash"] != 0]
rare_other = set(k1[(k1["kind"] != "question") & (k1["key1_count"] == 0)]["id"])
qcount = k1[k1["kind"] == "question"].set_index("id")["key1_count"].to_dict()
dropped = k1["kind"].eq("question") & k1["id"].map(lambda i: qcount.get(i, -1) > 10 and i in rare_other)
kept = k1[~dropped.to_numpy()]
t1 = KeyTable(k1.key1_hash.to_numpy(np.uint64), k1.id.to_numpy(object))
t2 = KeyTable(k2.key2_hash.to_numpy(np.uint64), k2.id.to_numpy(object))
t1v2 = KeyTable(kept.key1_hash.to_numpy(np.uint64), kept.id.to_numpy(object))
print(f"items with key2: {k2.id.nunique():,} of {k1.id.nunique():,} keyed", flush=True)
cache = {}


def keyset(item):
    if item not in cache:
        n = number.get(item)
        rows = np.nonzero((index.holders == n) & keep)[0] if n is not None else np.empty(0, dtype=np.int64)
        cache[item] = index.hashes[slot_of_row[rows]]
    return cache[item]


def hits(item, hashes):
    a = item in set(t1.items_of(t1.found(hashes)))
    b = a or item in set(t2.items_of(t2.found(hashes)))
    ksh = keyset(item)
    c = item in set(t1v2.items_of(t1v2.found(hashes))) or (
        item not in rare_other and len(ksh) > 0 and int(np.isin(ksh, hashes).sum()) >= min(3, len(ksh))
    )
    return a, b, c


# 1. Planted edits.
rng = random.Random(1)
KEYED = set(k1.id)
items = []
for path in sorted(Path(f"{S1}/items").glob("*.parquet")):
    rows = [
        r for r in pq.read_table(path, columns=["id", "eval", "question", "choices"]).to_pylist() if r["id"] in KEYED
    ]
    items += rng.sample(rows, min(8, len(rows)))
VOCAB = "apple river seven blue market yesterday garden silver eleven window".split()


def render(q, choices):
    return q + ("\n" + "\n".join(f"{'ABCDEFGHIJ'[i]}. {c}" for i, c in enumerate(choices[:10])) if choices else "")


def subst(t, n):
    w = t.split()
    for _ in range(n):
        if w:
            w[rng.randrange(len(w))] = rng.choice(VOCAB)
    return " ".join(w)


def typo(t, n):
    w = t.split()
    long = [i for i, x in enumerate(w) if len(x) >= 4]
    for i in rng.sample(long, min(n, len(long))):
        j = rng.randrange(len(w[i]) - 1)
        x = w[i]
        w[i] = x[:j] + x[j + 1] + x[j] + x[j + 2 :]
    return " ".join(w)


def numbers(t):
    return re.sub(r"\d+", lambda m: str(int(m.group()) + 1), t)


def delete(t, frac):
    return " ".join(x for x in t.split() if rng.random() > frac)


def reorder(item):
    c = list(item["choices"] or [])
    rng.shuffle(c)
    return render(item["question"], c)


FORMS = {
    "plain": lambda it: render(it["question"], it["choices"] or []),
    "typo in 1 word": lambda it: typo(render(it["question"], it["choices"] or []), 1),
    "typos in 3 words": lambda it: typo(render(it["question"], it["choices"] or []), 3),
    "1 word changed": lambda it: subst(render(it["question"], it["choices"] or []), 1),
    "3 words changed": lambda it: subst(render(it["question"], it["choices"] or []), 3),
    "every number +1": lambda it: numbers(render(it["question"], it["choices"] or [])),
    "choices reordered": reorder,
    "10% of words deleted": lambda it: delete(render(it["question"], it["choices"] or []), 0.1),
}
host = "The committee met on Tuesday to review the budget and the new library hours. " * 40
res = defaultdict(lambda: np.zeros(4))
has_digits = 0
for it in items:
    for name, form in FORMS.items():
        text = form(it)
        if name == "every number +1" and not re.search(r"\d", render(it["question"], it["choices"] or [])):
            continue
        if name == "choices reordered" and len(it["choices"] or []) < 2:
            continue
        hashes = np.concatenate([r.hashes for r in record_readings(host + "\n" + text + "\n" + host)])
        a, b, c = hits(it["id"], hashes)
        res[name] += (a, b, c, 1)
print(f"\n1. planted edits ({len(items)} items, {len({i['eval'] for i in items})} evals)", flush=True)
print(f"{'form':22s} {'n':>5s} {'key1':>6s} {'key1|key2':>10s} {'proposed':>9s}")
for name, (a, b, c, n) in res.items():
    print(f"{name:22s} {int(n):5d} {a / n:6.3f} {b / n:10.3f} {c / n:9.3f}")

# 2. key2's pair cost on 2 local shards.
records = []
for f in sorted(glob.glob(f"{S1}/corpus/*.parquet")):
    records += [
        r["text"]
        for r in pq.read_table(f, columns=["text"]).to_pylist()
        if r["text"] and len(r["text"]) <= MAX_DOC_CHARS
    ]
sample = records[:: max(1, len(records) // 60000)]
p1 = p12 = 0
for text in sample:
    hashes = np.concatenate([r.hashes for r in record_readings(text)])
    s1 = set(t1.items_of(t1.found(hashes)))
    s2 = set(t2.items_of(t2.found(hashes)))
    p1 += len(s1)
    p12 += len(s1 | s2)
print(
    f"\n2. key2 pair cost on {len(sample):,} records of 2 shards: key1 {p1:,} pairs, key1|key2 {p12:,} ({p12 / max(p1, 1):.2f}x)",
    flush=True,
)

# 3. Key-1 recall on near-certain copies (all 8 spread windows held) by the item's median window count in the reference.
frame = pd.DataFrame({"holder": index.holders, "count": total[slot_of_row]})
median = frame.groupby("holder")["count"].median()
names = [p.name for p in StoragePath(f"{B}/keyset-spread-1/rows/*.parquet").glob()]
with ThreadPoolExecutor(32) as pool:
    rows = pd.concat(
        list(
            pool.map(
                lambda n: pq.read_table(
                    io.BytesIO(StoragePath(f"{B}/keyset-spread-1/rows/{n}").read_bytes())
                ).to_pandas(),
                names,
            )
        ),
        ignore_index=True,
    )
copies = rows[(rows["key_set"] == 8) & (rows["held"] == 8)].copy()
copies["median"] = copies["item"].map(lambda i: median.get(number.get(i), np.nan))
copies["bucket"] = pd.cut(copies["median"], [-0.5, 0.5, 10, 100, 1e12], labels=["0", "1-10", "11-100", ">100"])
print(
    f"\n3. near-certain copies (all 8 of 8 spread windows held): {len(copies):,} pairs, {copies.item.nunique():,} items"
)
print(
    copies.groupby("bucket", observed=True)
    .agg(pairs=("key1", "size"), items=("item", "nunique"), key1_recall=("key1", "mean"))
    .to_string()
)
print(round(time.time() - T0), "s total")
