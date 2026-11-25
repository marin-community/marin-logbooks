# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""E5: planted edited copies; recall of key 1, key 1 or 2, dense share >= 0.5, and >= 2 indexed windows."""

import io
import random
import sys
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from marin.datakit.decon_keys import KeyTable, record_readings
from rigging.filesystem.storage_path import StoragePath

sys.path.insert(0, ".")
from experiments.datakit.decontam.encoder.dense_survey import _matched, dense_index

S1 = "work"
exec(open(f"{S1}/e1_planted.py").read().split("HOMO = ")[0].split("locators = pq.read_table")[0])
locators = pq.read_table(f"{S1}/locators.parquet")
index = dense_index(f"{S1}/locators.parquet")
number = {item: n for n, item in enumerate(index.items)}
keys = pq.read_table(
    io.BytesIO(
        StoragePath(
            "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc/decon-poc/stage1-v11/keys.parquet"
        ).read_bytes()
    ),
    columns=["id", "key1_hash", "key2_hash"],
).to_pandas()
k1 = keys[keys.key1_hash != 0]
k2 = keys[keys.key2_hash != 0]
t1 = KeyTable(k1.key1_hash.to_numpy(np.uint64), k1.id.to_numpy(object))
t12 = KeyTable(
    np.concatenate([k1.key1_hash.to_numpy(np.uint64), k2.key2_hash.to_numpy(np.uint64)]),
    np.concatenate([k1.id.to_numpy(object), k2.id.to_numpy(object)]),
)
rng = random.Random(1)
keyed = set(k1.id)
items = []
for path in sorted(__import__("pathlib").Path(f"{S1}/items").glob("*.parquet")):
    rows = [
        r
        for r in pq.read_table(path, columns=["id", "eval", "question", "choices"]).to_pylist()
        if r["id"] in keyed and r["id"] in number and index.windows_of[number[r["id"]]] >= 4
    ]
    items += rng.sample(rows, min(8, len(rows)))
VOCAB = "apple river seven blue market yesterday garden silver eleven window".split()


def edit(text, n):
    words = text.split()
    for _ in range(n):
        if words:
            words[rng.randrange(len(words))] = rng.choice(VOCAB)
    return " ".join(words)


def delete(text, frac):
    words = text.split()
    keep = [w for w in words if rng.random() > frac]
    return " ".join(keep)


def reorder(text):
    parts = [p for p in text.replace("?", ".").split(".") if p.strip()]
    rng.shuffle(parts)
    return ". ".join(parts)


FORMS = {
    "exact": lambda t: t,
    "subst1": lambda t: edit(t, 1),
    "subst2": lambda t: edit(t, 2),
    "subst3": lambda t: edit(t, 3),
    "subst5": lambda t: edit(t, 5),
    "delete10%": lambda t: delete(t, 0.10),
    "delete25%": lambda t: delete(t, 0.25),
    "reorder": reorder,
}
host = "The committee met on Tuesday to review the budget and the new library hours. " * 40
hits = defaultdict(int)
for item in items:
    n = number[item["id"]]
    for name, form in FORMS.items():
        text = host + "\n" + form(item["question"]) + "\n" + host
        hashes = np.concatenate([r.hashes for r in record_readings(text)])
        hits[(name, "key1")] += item["id"] in set(t1.items_of(t1.found(hashes)))
        hits[(name, "key1|key2")] += item["id"] in set(t12.items_of(t12.found(hashes)))
        m = _matched(index, hashes).get(n, 0)
        hits[(name, "dense>=0.5")] += m >= 0.5 * index.windows_of[n]
        hits[(name, ">=2 windows")] += m >= 2
print(len(items), "items with >= 4 indexed windows")
rules = ["key1", "key1|key2", "dense>=0.5", ">=2 windows"]
print(f"{'form':12s}" + "".join(f"{r:>13s}" for r in rules))
for name in FORMS:
    print(f"{name:12s}" + "".join(f"{hits[(name, r)] / len(items):>13.3f}" for r in rules))
