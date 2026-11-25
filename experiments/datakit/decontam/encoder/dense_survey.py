# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the stage-1 key with a dense wide rule on real normalized shards.

The dense rule indexes every ``STRIDE``-th 20-character window of each item's located question stream (windows
that more than ``MAX_HOLDERS`` items share are dropped). A record is a dense candidate for an item when its
readings hold at least ``MIN_SHARE`` of the item's indexed windows, in any order. A key candidate is an item
whose key-1 window a reading holds (the pipeline's stage 1, without excerpts). Dense-only pairs are written
with an excerpt around their first matched window, to be read.

    python -m experiments.datakit.decontam.encoder.dense_survey --locators <locators.parquet> \\
        --sources experiments/datakit/decontam/encoder/sources.json --input-root <normalized root> \\
        --per-source 1 --output <dir>
"""

import argparse
import functools
import json
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.decon_keys import MAX_DOC_CHARS, WINDOW, KeyTable, record_readings, window_hashes
from marin.datakit.decon_tables import read_table, write_table
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_file

from experiments.datakit.decontam.encoder.encoding_survey import survey_shards

logger = logging.getLogger(__name__)

STRIDE = 5
MAX_HOLDERS = 20
MIN_SHARE = 0.5
BEFORE, AFTER = 400, 1600
ROWS_PER_SHARD = 400


@dataclass(frozen=True)
class DenseIndex:
    hashes: np.ndarray  # sorted unique window hashes
    starts: np.ndarray  # item rows of hash i: holders[starts[i]:starts[i + 1]]
    holders: np.ndarray
    positions: np.ndarray  # stream offset of each holder row's window in its item's question
    windows_of: np.ndarray  # indexed windows per item
    items: list[str]
    evals: list[str]
    key_table: KeyTable
    key_items: np.ndarray  # item index per key-table row


@functools.lru_cache(maxsize=1)
def dense_index(locators_path: str) -> DenseIndex:
    locators = read_table(locators_path, ["item", "kind", "eval", "key1_hash", "question"]).to_pydict()
    items = sorted(set(locators["item"]))
    number = {item: n for n, item in enumerate(items)}
    evals = [""] * len(items)
    hash_parts, owner_parts, position_parts = [], [], []
    seen: set[str] = set()
    for item, eval_name, question in zip(locators["item"], locators["eval"], locators["question"], strict=True):
        evals[number[item]] = eval_name
        if item in seen or not question:
            continue
        seen.add(item)
        hashes, first = np.unique(window_hashes(question, WINDOW)[::STRIDE], return_index=True)
        hash_parts.append(hashes)
        owner_parts.append(np.full(len(hashes), number[item], dtype=np.int32))
        position_parts.append((first * STRIDE).astype(np.int32))
    hashes = np.concatenate(hash_parts)
    owners = np.concatenate(owner_parts)
    positions = np.concatenate(position_parts)
    order = np.argsort(hashes, kind="stable")
    hashes, owners, positions = hashes[order], owners[order], positions[order]
    unique, starts, counts = np.unique(hashes, return_index=True, return_counts=True)
    keep = counts <= MAX_HOLDERS
    keep_rows = np.repeat(keep, counts)
    hashes, owners, positions = hashes[keep_rows], owners[keep_rows], positions[keep_rows]
    unique, starts = np.unique(hashes, return_index=True)
    starts = np.append(starts, len(hashes))
    windows_of = np.bincount(owners, minlength=len(items))
    key_table = KeyTable(np.asarray(locators["key1_hash"], dtype=np.uint64), np.arange(len(locators["item"])))
    key_items = np.array([number[item] for item in locators["item"]], dtype=np.int32)
    logger.info("dense index: %d windows, %d items with windows", len(hashes), int((windows_of > 0).sum()))
    return DenseIndex(unique, starts, owners, positions, windows_of, items, evals, key_table, key_items)


def _matched(index: DenseIndex, hashes: np.ndarray) -> dict[int, int]:
    """Item -> number of its indexed windows among *hashes*."""
    hashes = np.unique(hashes)
    slot = np.minimum(np.searchsorted(index.hashes, hashes), len(index.hashes) - 1)
    slots = slot[index.hashes[slot] == hashes]
    if not len(slots):
        return {}
    owners = np.concatenate([index.holders[index.starts[s] : index.starts[s + 1]] for s in slots.tolist()])
    values, counts = np.unique(owners, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist(), strict=True))


def make_dense(locators_path: str, output: str):
    def dense(paths: Iterator[str], shard: ShardInfo) -> Iterator[dict]:
        index = dense_index(locators_path)
        for path in paths:
            name = path.split("/normalized/", 1)[-1].replace("/", "__")
            done = StoragePath(prefix_join(output, f"stats/{name}.json"))
            if done.exists():
                yield json.loads(done.read_text())
                continue
            stats: Counter = Counter()
            rows: list[dict] = []
            for record in load_file(InputFileSpec(path=path, columns=["id", "text"])):
                text = str(record.get("text") or "")
                stats["records"] += 1
                if len(text) > MAX_DOC_CHARS:
                    stats["long_records"] += 1
                    continue
                readings_of = record_readings(text)
                all_hashes = np.concatenate([reading.hashes for reading in readings_of])
                key_hits = {
                    int(index.key_items[row]) for row in index.key_table.items_of(index.key_table.found(all_hashes))
                }
                matched = _matched(index, all_hashes)
                dense_hits = {
                    item
                    for item, count in matched.items()
                    if index.windows_of[item] >= 2 and count >= MIN_SHARE * index.windows_of[item]
                }
                stats["key_pairs"] += len(key_hits)
                stats["dense_pairs"] += len(dense_hits)
                stats["both"] += len(key_hits & dense_hits)
                stats["key_records"] += bool(key_hits)
                stats["dense_records"] += bool(dense_hits)
                for item in dense_hits - key_hits:
                    stats["dense_only"] += 1
                    if len(rows) >= ROWS_PER_SHARD:
                        continue
                    rows.append(_row(path, str(record["id"]), item, matched[item], index, readings_of))
                stats["key_only"] += len(key_hits - dense_hits)
            if rows:
                write_table(pa.Table.from_pylist(rows), prefix_join(output, f"dense_only/{name}.parquet"))
            result = {"path": path, **stats}
            done.write_text(json.dumps(result))
            yield result

    return dense


def _row(path: str, record_id: str, item: int, count: int, index: DenseIndex, readings_of) -> dict:
    """A dense-only pair, with the text around the first window of the item that a reading holds."""
    item_rows = np.nonzero(index.holders == item)[0]
    item_hashes = index.hashes[np.searchsorted(index.starts, item_rows, side="right") - 1]
    excerpt, reading_number = "", -1
    for number, reading in enumerate(readings_of):
        positions = np.nonzero(np.isin(reading.hashes, item_hashes))[0]
        if len(positions):
            offset = reading.text_offset(int(positions.min()))
            excerpt, reading_number = reading.text[max(0, offset - BEFORE) : offset + AFTER], number
            break
    return {
        "path": path,
        "record": record_id,
        "item": index.items[item],
        "eval": index.evals[item],
        "matched_windows": count,
        "item_windows": int(index.windows_of[item]),
        "reading": reading_number,
        "excerpt": excerpt,
    }


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locators", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--per-source", type=int, default=1)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-workers", type=int, default=160)
    parser.add_argument("--part", type=int, default=None, help="scan part --part of --parts in this process")
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--ram", default="24g")
    args = parser.parse_args()
    files = survey_shards(args.sources, args.input_root, args.per_source, args.seed)
    StoragePath(prefix_join(args.output, "shards.json")).write_text(json.dumps(files))
    if args.part is not None:

        class _Shard:
            shard_idx = args.part

        mine = files[args.part :: args.parts]
        logger.info("part %d of %d: %d shards", args.part, args.parts, len(mine))
        for result in make_dense(args.locators, args.output)(iter(mine), _Shard()):
            logger.info("%s", {k: v for k, v in result.items() if isinstance(v, int)})
        return
    resources = ResourceConfig(cpu=1, ram=args.ram, preemptible=False)
    context = ZephyrContext(
        name="decon-dense-survey", resources=resources, max_workers=min(args.max_workers, len(files))
    )
    outcome = context.execute(
        Dataset.from_list(files).map_shard(make_dense(args.locators, args.output)), map_task_resources=resources
    )
    totals: Counter = Counter()
    for result in outcome.results:
        totals.update({k: v for k, v in result.items() if isinstance(v, int)})
    StoragePath(prefix_join(args.output, "totals.json")).write_text(json.dumps(dict(totals), indent=1))
    logger.info("dense survey done: %s", dict(totals))


if __name__ == "__main__":
    main()
