# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F8: a fixed regression set of known contamination (review point 7), and a stage-1 check against it.

``build`` joins, one row per (record, item) with the record text that holds the copy:
  - ``key1_miss``: copies stage 2 removes that key 1 never sent (dense survey, scored by iteration 16, MATH
    [asy] boilerplate dropped);
  - ``encoded``: copies that only a URL or HTML decoding finds (encoding survey, window share >= 0.8), with the
    raw record text;
  - ``judged``: pairs of label sets 19, 23, 26, 27 and 28 that both judges call COPY or EDITED (``label``), and
    the pairs both call clean (``label`` = 0), as stage-2 precision cases.
``check`` reports, per kind, the share of rows whose text holds the item's key 1 (as stage 1 reads it), and
key 1 or key 2 at count 0.

    python f8_regression.py build --output <parquet>
    python f8_regression.py check --regression <parquet> --keys <keys.parquet>
"""
import argparse
import io
import json
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from marin.datakit.decon_keys import KeyTable, record_readings
from rigging.filesystem.storage_path import StoragePath

WORK = "work"
LABELS_LOCAL = "/Users/rav/projects/marin-worktrees/pr-10836-analysis/label"
LABELS = "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc/decon-poc/labels"
SETS = ("set19", "set23", "set26", "set27", "set28")
GOOD, BAD = {"COPY", "EDITED"}, {"VARIANT", "GENERIC", "UNRELATED"}
JUDGES = ("glm-5.3", "deepseek-v4.1-flash")
BOILERPLATE = re.compile(r"rr_cartesian_axes|real ticklen=3")


def _labels(name: str, judge: str) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for path in StoragePath(f"{LABELS}/{name}/**/{judge}/batch_*.jsonl").glob()
        for line in path.read_text().split("\n")
        if line.strip()
    ]
    return pd.DataFrame(rows).drop_duplicates("uid").set_index("uid")["label"].rename(judge)


def build(output: str) -> None:
    misses = pd.read_parquet(f"{WORK}/dense_scored.parquet")
    misses = misses[(misses["score"] >= 0.6) & ~misses["right"].str.contains(BOILERPLATE)]
    frames = [
        misses.assign(kind="key1_miss", label=1, source="dense-survey-1")[
            ["path", "record", "item", "eval", "right", "kind", "label", "source"]
        ]
    ]
    encoded = pd.DataFrame(json.load(open(f"{WORK}/e2_new_scored.json")))
    encoded = encoded[encoded["share"].astype(float) >= 0.8]
    raw = {}
    for path, group in encoded.groupby("path"):
        table = pq.read_table(io.BytesIO(StoragePath(path).read_bytes()), columns=["id", "text"]).to_pandas()
        raw.update(table[table["id"].astype(str).isin(set(group["record"]))].set_index("id")["text"].to_dict())
    encoded["right"] = [raw[r] for r in encoded["record"]]
    frames.append(
        encoded.assign(kind=lambda f: "encoded_" + f["decoding"], label=1, source="encoding-survey-2")[
            ["path", "record", "item", "eval", "right", "kind", "label", "source"]
        ]
    )
    for name in SETS:
        pairs = pd.read_parquet(f"{LABELS_LOCAL}/{name}/pairs.parquet").set_index("uid")
        joined = pairs.join(_labels(name, JUDGES[0]), how="inner").join(_labels(name, JUDGES[1]), how="inner")
        a, b = joined[JUDGES[0]], joined[JUDGES[1]]
        good, bad = a.isin(GOOD) & b.isin(GOOD), a.isin(BAD) & b.isin(BAD)
        both_copy = ((a == "COPY") & (b == "COPY"))[good | bad]
        part = joined[good | bad].assign(label=good[good | bad].astype(int), source=name)
        part["kind"] = np.where(part["label"] == 1, np.where(both_copy, "judged_copy", "judged_edited"), "judged_clean")
        part["path"] = part["path"] if "path" in part else ""
        frames.append(part.reset_index()[["path", "record", "item", "eval", "right", "kind", "label", "source"]])
    regression = pd.concat(frames, ignore_index=True).drop_duplicates(["record", "item", "kind"])
    regression.to_parquet(output)
    print(regression.groupby(["kind", "label"]).size().to_string())


def _proposed(locators: str, counts: str, cap: int, keys: pd.DataFrame):
    """A test for the proposed stage 1: key 1 without the dropped question keys, or at least 3 of the item's
    spread key set (windows held at most *cap* times) for an item without a count-0 passage or choices key."""
    from experiments.datakit.decontam.encoder import key_set_survey as ks

    index, _total, keep = ks.key_sets(locators, counts, "spread", cap)
    slot_of_row = np.searchsorted(index.starts, np.arange(len(index.holders)), side="right") - 1
    number = {item: n for n, item in enumerate(index.items)}
    k1 = keys[keys["key1_hash"] != 0]
    rare_other = set(k1[(k1["kind"] != "question") & (k1["key1_count"] == 0)]["id"])
    qcount = k1[k1["kind"] == "question"].set_index("id")["key1_count"].to_dict()
    dropped = k1["kind"].eq("question") & k1["id"].map(lambda i: qcount.get(i, -1) > 10 and i in rare_other)
    kept = k1[~dropped.to_numpy()]
    table = KeyTable(kept["key1_hash"].to_numpy(np.uint64), kept["id"].to_numpy(object))
    cache = {}

    def holds(item: str, hashes: np.ndarray) -> bool:
        if item in set(table.items_of(table.found(hashes))):
            return True
        if item in rare_other or item not in number:
            return False
        if item not in cache:
            rows = np.nonzero((index.holders == number[item]) & keep)[0]
            cache[item] = index.hashes[slot_of_row[rows]]
        key_set = cache[item]
        return len(key_set) > 0 and int(np.isin(key_set, hashes).sum()) >= min(3, len(key_set))

    return holds


def check(
    regression_path: str, keys_path: str, locators: str | None = None, counts: str | None = None, cap: int = 100
) -> None:
    regression = pd.read_parquet(regression_path)
    keys = pq.read_table(
        io.BytesIO(StoragePath(keys_path).read_bytes()) if keys_path.startswith("s3://") else keys_path
    ).to_pandas()
    k1 = keys[keys["key1_hash"] != 0]
    k2 = keys[(keys["key2_hash"] != 0) & (keys["key2_count"] == 0)]
    t1 = KeyTable(k1["key1_hash"].to_numpy(np.uint64), k1["id"].to_numpy(object))
    t2 = KeyTable(k2["key2_hash"].to_numpy(np.uint64), k2["id"].to_numpy(object))
    proposed = _proposed(locators, counts, cap, keys) if locators else None
    hit1, hit12, hitp = [], [], []
    for text, item in zip(regression["right"], regression["item"]):
        hashes = np.concatenate([r.hashes for r in record_readings(text or "")])
        one = item in set(t1.items_of(t1.found(hashes)))
        hit1.append(one)
        hit12.append(one or item in set(t2.items_of(t2.found(hashes))))
        hitp.append(proposed(item, hashes) if proposed else False)
    regression["key1"], regression["key1_or_key2"], regression["proposed"] = hit1, hit12, hitp
    columns = ["key1", "key1_or_key2"] + (["proposed"] if proposed else [])
    print(
        regression.groupby(["kind", "label"])[columns]
        .mean()
        .round(3)
        .assign(rows=regression.groupby(["kind", "label"]).size())
        .to_string()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build", "check"])
    parser.add_argument("--output")
    parser.add_argument("--regression")
    parser.add_argument("--keys")
    parser.add_argument("--locators", help="check: also test the proposed stage 1 (needs --counts)")
    parser.add_argument("--counts", help="check: the key-set count directory (counts/*.npz)")
    parser.add_argument("--cap", type=int, default=100)
    args = parser.parse_args()
    if args.mode == "build":
        build(args.output)
    else:
        check(args.regression, args.keys, args.locators, args.counts, args.cap)


if __name__ == "__main__":
    main()
