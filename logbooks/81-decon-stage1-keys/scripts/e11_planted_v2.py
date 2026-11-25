# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""E11: planted forms and planted edits under the proposed stage 1 (key 1 minus common question keys, plus a
spread key set capped at 100, >= 3 of 8) against today's key 1."""

import io
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from marin.datakit.decon_keys import KeyTable, record_readings
from rigging.filesystem.storage_path import StoragePath

sys.path.insert(0, ".")
from experiments.datakit.decontam.encoder import key_set_survey as ks

S1 = "work"
src = open(f"{S1}/e1_planted.py").read()
exec(src.split("HOMO = ")[0].split("locators = pq.read_table")[0])  # imports, text_of
exec("HOMO = " + src.split("HOMO = ")[1].split("host = (")[0])  # FORMS


def text_of(item):
    choices = item["choices"] or []
    return item["question"] + (
        "\n" + "\n".join(f"{'ABCDEFGHIJ'[k]}. {c}" for k, c in enumerate(choices[:10])) if choices else ""
    )


import time

T0 = time.time()
index, total, keep = ks.key_sets(
    f"{S1}/locators.parquet", "s3://marin-us-east-08a/tmp/ttl=7d/rav/stage1-explore/keyset-count-1/counts", "spread", 100
)
number = {item: n for n, item in enumerate(index.items)}
print("key sets built", round(time.time() - T0), "s", flush=True)
key_size = np.bincount(index.holders[keep], minlength=len(index.items))
k = pq.read_table(
    io.BytesIO(
        StoragePath(
            "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc/decon-poc/stage1-v11/keys.parquet"
        ).read_bytes()
    )
).to_pandas()
k = k[k["key1_hash"] != 0]
rare_other = set(k[(k["kind"] != "question") & (k["key1_count"] == 0)]["id"])
qcount = k[k["kind"] == "question"].set_index("id")["key1_count"].to_dict()
dropped = k["kind"].eq("question") & k["id"].map(lambda i: qcount.get(i, -1) > 10 and i in rare_other)
kept = k[~dropped.to_numpy()]
t1 = KeyTable(k.key1_hash.to_numpy(np.uint64), k.id.to_numpy(object))
t1v2 = KeyTable(kept.key1_hash.to_numpy(np.uint64), kept.id.to_numpy(object))
slot_of_row = np.searchsorted(index.starts, np.arange(len(index.holders)), side="right") - 1
_keyset_cache = {}


def keyset_hashes(item):
    if item not in _keyset_cache:
        n = number.get(item)
        rows = np.nonzero((index.holders == n) & keep)[0] if n is not None else np.empty(0, dtype=np.int64)
        _keyset_cache[item] = index.hashes[slot_of_row[rows]]
    return _keyset_cache[item]


def held(item, hashes):
    ks_h = keyset_hashes(item)
    return int(np.isin(ks_h, hashes).sum()), len(ks_h)


rng = random.Random(1)
KEYED = set(k.id)
items = []
for path in sorted(Path(f"{S1}/items").glob("*.parquet")):
    rows = [
        r for r in pq.read_table(path, columns=["id", "eval", "question", "choices"]).to_pylist() if r["id"] in KEYED
    ]
    items += rng.sample(rows, min(8, len(rows)))
VOCAB = "apple river seven blue market yesterday garden silver eleven window".split()


def edit(t, n):
    w = t.split()
    for _ in range(n):
        if w:
            w[rng.randrange(len(w))] = rng.choice(VOCAB)
    return " ".join(w)


def delete(t, frac):
    return " ".join(x for x in t.split() if rng.random() > frac)


EDITS = {
    "subst1": lambda t: edit(t, 1),
    "subst3": lambda t: edit(t, 3),
    "subst5": lambda t: edit(t, 5),
    "delete10%": lambda t: delete(t, 0.1),
    "delete25%": lambda t: delete(t, 0.25),
}
ALL = {
    **{
        f: FORMS[f]
        for f in (
            "plain",
            "json_twice",
            "markdown_escapes",
            "html_escape",
            "line_numbers",
            "url_plus",
            "homoglyphs",
            "word_per_line",
        )
    },
    **EDITS,
}
host = "The committee met on Tuesday to review the budget and the new library hours. " * 40
res = defaultdict(lambda: [0, 0, 0])
for number_done, item in enumerate(items):
    if number_done % 25 == 0:
        print("item", number_done, round(time.time() - T0), "s", flush=True)
    text = text_of(item)
    for name, form in ALL.items():
        rec = host + "\n" + form(text) + "\n" + host
        hashes = np.concatenate([r.hashes for r in record_readings(rec)])
        a = item["id"] in set(t1.items_of(t1.found(hashes)))
        b1 = item["id"] in set(t1v2.items_of(t1v2.found(hashes)))
        h, size = held(item["id"], hashes)
        b2 = item["id"] not in rare_other and size > 0 and h >= min(3, size)
        res[name][0] += a
        res[name][1] += b1 or b2
        res[name][2] += 1
print(len(items), "planted items from", len({i["eval"] for i in items}), "evals")
print(f"{'form':18s} {'key 1 today':>12s} {'proposed':>10s}")
for name, (a, b, n) in res.items():
    print(f"{name:18s} {a / n:12.3f} {b / n:10.3f}")
