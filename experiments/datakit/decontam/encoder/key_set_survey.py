# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure a k-of-n key rule for stage 1: each item's ``KEY_SET`` rarest question windows, matched k at a time.

``count``: over one sample of shards, the number of records that hold each window of the dense index
(``dense_survey.dense_index``: every 5th window of each item's located question stream). ``scan``: over a
second sample, each item's key set is its ``KEY_SET`` windows with the lowest summed counts; every record that
holds a key-set window is written with, per item, the number of key-set windows it holds and whether it holds
the item's production key 1. Load and recall of "at least k of the key set" are computed offline from them.

    python -m experiments.datakit.decontam.encoder.key_set_survey count|scan --locators <locators.parquet> \\
        --sources experiments/datakit/decontam/encoder/sources.json --input-root <normalized root> \\
        --seed <seed> --output <dir> --part K --parts N
"""

import argparse
import functools
import io
import json
import logging
from collections import Counter

import numpy as np
import pyarrow as pa
from marin.datakit.decon_keys import MAX_DOC_CHARS, WINDOW, excerpt, record_readings
from marin.datakit.decon_tables import read_frame, read_table, write_table
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_file

from experiments.datakit.decontam.encoder.dense_survey import DenseIndex, dense_index
from experiments.datakit.decontam.encoder.encoding_survey import survey_shards

logger = logging.getLogger(__name__)

KEY_SET = 8
# How a key set is chosen: the KEY_SET rarest windows of the item, or the rarest window of each of KEY_SET equal
# segments of the item's question (spread over the text, so a junction cannot hold the whole set).
SELECTIONS = ("rarest", "spread")


def _name(path: str) -> str:
    return path.split("/normalized/", 1)[-1].replace("/", "__")


def _slots(index: DenseIndex, hashes: np.ndarray) -> np.ndarray:
    hashes = np.unique(hashes)
    slot = np.minimum(np.searchsorted(index.hashes, hashes), len(index.hashes) - 1)
    return slot[index.hashes[slot] == hashes]


def count_shard(index: DenseIndex, path: str, output: str, exclude_copies: int | None = None) -> dict:
    """Records per dense-index window. With *exclude_copies*, a record that holds at least that many of an item's
    windows (a likely copy) does not count toward that item's windows, so a count measures how generic a window
    is rather than how often its item is copied."""
    target = StoragePath(prefix_join(output, f"counts/{_name(path)}.npz"))
    if target.exists():
        return {"path": path, "skipped": 1}
    counts = np.zeros(len(index.hashes), dtype=np.uint32)
    records = skipped = 0
    for record in load_file(InputFileSpec(path=path, columns=["text"])):
        text = str(record.get("text") or "")
        if len(text) > MAX_DOC_CHARS:
            continue
        records += 1
        slots = _slots(index, np.concatenate([reading.hashes for reading in record_readings(text)]))
        if exclude_copies is not None and len(slots):
            rows = np.concatenate([np.arange(index.starts[s], index.starts[s + 1]) for s in slots.tolist()])
            holders = index.holders[rows]
            items, held = np.unique(holders, return_counts=True)
            copied = items[held >= exclude_copies]
            if len(copied):
                row_slots = np.repeat(slots, np.diff(index.starts)[slots])
                dropped = np.unique(row_slots[np.isin(holders, copied)])
                skipped += len(dropped)
                slots = np.setdiff1d(slots, dropped)
        counts[slots] += 1
    nonzero = np.nonzero(counts)[0]
    buffer = io.BytesIO()
    np.savez(buffer, slots=nonzero.astype(np.int64), counts=counts[nonzero])
    target.write_bytes(buffer.getvalue())
    return {"path": path, "records": records, "windows_seen": len(nonzero), "copy_windows_skipped": skipped}


@functools.lru_cache(maxsize=1)
def key_sets(
    locators_path: str, counts_dir: str, selection: str = "rarest", cap: int | None = None
) -> tuple[DenseIndex, np.ndarray, np.ndarray]:
    """The dense index, each slot's summed count, and for each slot a mask of the items whose key set holds it
    (as a slot -> item mapping in the index's holder rows)."""

    index = dense_index(locators_path)
    total = np.zeros(len(index.hashes), dtype=np.int64)
    for path in StoragePath(prefix_join(counts_dir, "*.npz")).glob():
        part = np.load(io.BytesIO(path.read_bytes()))
        np.add.at(total, part["slots"], part["counts"])
    # Holder rows: (slot, item); keep, per item, the KEY_SET rows with the lowest slot count.
    slot_of_row = np.searchsorted(index.starts, np.arange(len(index.holders)), side="right") - 1
    keep_rows = np.zeros(len(index.holders), dtype=bool)
    if selection == "rarest":
        order = np.lexsort((total[slot_of_row], index.holders))
        items_sorted = index.holders[order]
        first = np.searchsorted(items_sorted, items_sorted, side="left")
        rank = np.arange(len(order)) - first
        keep_rows[order[rank < KEY_SET]] = True
    else:
        by_position = np.lexsort((index.positions, index.holders))
        items_sorted = index.holders[by_position]
        first = np.searchsorted(items_sorted, items_sorted, side="left")
        rows_of = np.bincount(index.holders, minlength=len(index.items))
        rank = np.arange(len(by_position)) - first
        segment = np.zeros(len(index.holders), dtype=np.int64)
        segment[by_position] = rank * KEY_SET // np.maximum(rows_of[items_sorted], 1)
        order = np.lexsort((index.positions, total[slot_of_row], segment, index.holders))
        key = index.holders[order].astype(np.int64) * (KEY_SET + 1) + segment[order]
        keep_rows[order[np.r_[True, key[1:] != key[:-1]]]] = True
    if cap is not None:
        # A window that the reference sample holds more than *cap* times (shared template text) is not a key.
        keep_rows &= total[slot_of_row] <= cap
    logger.info("key sets: %d rows of %d", int(keep_rows.sum()), len(keep_rows))
    return index, total, keep_rows


@functools.lru_cache(maxsize=1)
def _questions(locators_path: str) -> dict[str, tuple[str, int]]:
    """Item -> its located question stream and question length, as the scan's ``excerpt`` takes them."""
    locators = read_table(locators_path, ["item", "kind", "question", "question_chars"]).to_pydict()
    questions: dict[str, tuple[str, int]] = {}
    for item, kind, question, chars in zip(
        locators["item"], locators["kind"], locators["question"], locators["question_chars"], strict=True
    ):
        if kind == "question" and question:
            questions.setdefault(item, (question, int(chars)))
    return questions


@functools.lru_cache(maxsize=1)
def _rare_other(keys_path: str) -> frozenset[str]:
    """Items with a passage or choices key that no reference record holds: the key set is not added for them."""
    keys = read_frame(keys_path, ["id", "kind", "key1_hash", "key1_count"])
    rare = keys[(keys["kind"] != "question") & (keys["key1_hash"] != 0) & (keys["key1_count"] == 0)]
    return frozenset(rare["id"])


def scan_shard(
    locators_path: str,
    counts_dir: str,
    path: str,
    output: str,
    selection: str = "rarest",
    cap: int | None = None,
    keys_path: str | None = None,
    min_held: int = 3,
) -> dict:
    """Key-set hits of one shard. With *keys_path*, also writes the pairs that only the key set admits (the
    add-on: an item without a count-0 passage or choices key, at least *min_held* of its key set held, no key-1
    hit) with the excerpt the production scan would cut, to ``pairs/<name>.parquet`` for scoring."""
    target = StoragePath(prefix_join(output, f"stats/{_name(path)}.json"))
    if target.exists():
        return json.loads(target.read_text())
    index, _total, keep_rows = key_sets(locators_path, counts_dir, selection, cap)
    key_size = np.bincount(index.holders[keep_rows], minlength=len(index.items))
    rare_other = _rare_other(keys_path) if keys_path else frozenset()
    questions = _questions(locators_path) if keys_path else {}
    stats: Counter = Counter()
    rows: list[dict] = []
    pairs: list[dict] = []
    for record in load_file(InputFileSpec(path=path, columns=["id", "text"])):
        text = str(record.get("text") or "")
        stats["records"] += 1
        if len(text) > MAX_DOC_CHARS:
            continue
        readings_of = record_readings(text)
        hashes = np.concatenate([reading.hashes for reading in readings_of])
        key1 = {int(index.key_items[row]) for row in index.key_table.items_of(index.key_table.found(hashes))}
        slots = _slots(index, hashes)
        held: Counter = Counter()
        first_position: dict[int, int] = {}
        for slot in slots.tolist():
            for row in range(index.starts[slot], index.starts[slot + 1]):
                if keep_rows[row]:
                    holder = int(index.holders[row])
                    held[holder] += 1
                    first_position[holder] = min(first_position.get(holder, 1 << 30), int(index.positions[row]))
        if keys_path:
            for item, n in held.items():
                name = index.items[item]
                if item in key1 or name in rare_other or name not in questions or n < min(min_held, key_size[item]):
                    continue
                question, chars = questions[name]
                at = first_position[item]
                cut = excerpt(readings_of, chars, question, question[at : at + WINDOW], at)
                stats["addon_pairs"] += 1
                pairs.append(
                    {
                        "record": str(record["id"]),
                        "item": name,
                        "eval": index.evals[item],
                        "right": cut.text,
                        "held": n,
                        "key_set": int(key_size[item]),
                        "located": cut.at >= 0,
                        "record_chars": len(text),
                        "path": path,
                    }
                )
        stats["key1_records"] += bool(key1)
        stats["key1_pairs"] += len(key1)
        for k in (1, 2, 3):
            matched = {item for item, n in held.items() if n >= min(k, key_size[item])}
            stats[f"k{k}_pairs"] += len(matched)
            stats[f"k{k}_records"] += bool(matched)
            stats[f"k{k}_and_key1"] += len(matched & key1)
        for item, n in held.items():
            if n >= 2 or item in key1:
                rows.append(
                    {
                        "record": str(record["id"]),
                        "item": index.items[item],
                        "eval": index.evals[item],
                        "held": n,
                        "key_set": int(key_size[item]),
                        "key1": item in key1,
                    }
                )
        for item in key1 - set(held):
            rows.append(
                {
                    "record": str(record["id"]),
                    "item": index.items[item],
                    "eval": index.evals[item],
                    "held": 0,
                    "key_set": int(key_size[item]),
                    "key1": True,
                }
            )
    if rows:
        write_table(pa.Table.from_pylist(rows), prefix_join(output, f"rows/{_name(path)}.parquet"))
    if pairs:
        write_table(pa.Table.from_pylist(pairs), prefix_join(output, f"pairs/{_name(path)}.parquet"))
    result = {"path": path, **stats}
    target.write_text(json.dumps(result))
    return result


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["count", "scan"])
    parser.add_argument("--locators", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--selection", choices=SELECTIONS, default="rarest")
    parser.add_argument("--cap", type=int, default=None, help="drop key-set windows held more often in the reference")
    parser.add_argument("--counts", help="scan: the count output directory (holds counts/*.npz)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--part", type=int, required=True)
    parser.add_argument("--parts", type=int, required=True)
    parser.add_argument("--shards", help="a JSON list of shards to use instead of the seeded sample")
    parser.add_argument("--keys", help="scan: keys.parquet; write the add-on pairs with excerpts")
    parser.add_argument("--min-held", type=int, default=3, help="scan: key-set windows a record must hold")
    parser.add_argument("--exclude-copies", type=int, help="count: skip an item's windows in its likely copies")
    args = parser.parse_args()
    if args.shards:
        shards = json.loads(StoragePath(args.shards).read_text())
    else:
        shards = survey_shards(args.sources, args.input_root, 1, args.seed)
    files = shards[args.part :: args.parts]
    logger.info("%s part %d of %d: %d shards", args.mode, args.part, args.parts, len(files))
    for path in files:
        if args.mode == "count":
            result = count_shard(dense_index(args.locators), path, args.output, args.exclude_copies)
        else:
            result = scan_shard(
                args.locators,
                prefix_join(args.counts, "counts"),
                path,
                args.output,
                args.selection,
                args.cap,
                args.keys,
                args.min_held,
            )
        logger.info("%s", {k: v for k, v in result.items() if isinstance(v, int)})


if __name__ == "__main__":
    main()
