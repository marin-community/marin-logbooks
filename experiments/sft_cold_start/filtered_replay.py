# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the long replay caches of the cold-start mix from documents that pass the junk filter.

The control caches take whole long documents from a cell's source shards in listed order until the cell's
token budget is crossed (``replay_subset.select_length_strata``). This does the same but skips documents
``junk_verdict`` flags, so a cell keeps its budget and its place in the mixture and only its content
changes. A cell whose listed shards run out continues into the cell's remaining shards in name order.
Short caches are untouched.

Each candidate is judged from eight 2,048-token windows, so a row of hundreds of millions of tokens costs
eight small reads to reject.
"""

import gzip
import json
import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import click
import numpy as np
import tensorstore as ts
import yaml
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.store.cache import write_levanter_cache
from levanter.store.jagged_array import JaggedArrayStore
from marin.datakit.junk_filter import junk_verdict
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.sft_cold_start.replay_subset import LENGTH_THRESHOLD

logger = logging.getLogger(__name__)

WINDOW_TOKENS = 2_048
INTERIOR_WINDOWS = 6
JUDGE_BATCH_DOCUMENTS = 64
LONG_CACHE_NAME = "gt_64k"
TRAIN_SUFFIX = "-train"


@dataclass(frozen=True)
class ReplayCell:
    name: str
    source: str
    shards: tuple[str, ...]
    long_tokens: int


@dataclass(frozen=True)
class KeptDocument:
    shard: str
    index: int
    start: int
    stop: int


def read_cells(path: str) -> list[ReplayCell]:
    """Train cells with a long budget, from the reproduction's replay preparation config."""
    with gzip.open(path, "rt") as handle:
        config = yaml.safe_load(handle)
    return [
        ReplayCell(cell["name"], cell["source"], tuple(cell["shards"]), cell["long_tokens"])
        for cell in config["cells"]
        if cell["name"].endswith(TRAIN_SUFFIX) and cell["long_tokens"] > 0
    ]


def window_starts(length: int) -> list[int]:
    """Head, six evenly spaced interior positions and tail; the whole document when it is short."""
    if length <= WINDOW_TOKENS * (INTERIOR_WINDOWS + 2):
        return [0]
    interior = np.linspace(WINDOW_TOKENS, length - 2 * WINDOW_TOKENS, INTERIOR_WINDOWS + 2)[1:-1]
    return [0, *(int(v) for v in interior), length - WINDOW_TOKENS]


def shard_order(cell: ReplayCell) -> list[str]:
    """The control's shards in their listed order, then the cell's other shards by name."""
    listed = list(cell.shards)
    present = sorted(p.name for p in StoragePath.parse(cell.source).ls() if p.name.startswith("part-"))
    return listed + [name for name in present if name not in set(listed)]


def select_kept_documents(cell: ReplayCell, tokenizer) -> tuple[list[KeptDocument], list[dict]]:
    """Take unflagged long documents in shard order until the cell's long budget is crossed."""
    kept: list[KeptDocument] = []
    audit: list[dict] = []
    remaining = cell.long_tokens
    for shard in shard_order(cell):
        if remaining <= 0:
            break
        store = JaggedArrayStore.open(
            str(StoragePath.parse(cell.source) / shard / "input_ids"), mode="r", dtype=np.int32
        )
        count = len(store)
        offsets = np.asarray(store.offsets[: count + 1].read().result())
        ends = offsets[1:].astype(np.int64)
        begins = np.concatenate((np.array([0], dtype=np.int64), ends[:-1]))
        candidates = np.flatnonzero(ends - begins > LENGTH_THRESHOLD)
        for batch_start in range(0, len(candidates), JUDGE_BATCH_DOCUMENTS):
            if remaining <= 0:
                break
            batch = candidates[batch_start : batch_start + JUDGE_BATCH_DOCUMENTS]
            plans = []
            for index in batch:
                begin, end = int(begins[index]), int(ends[index])
                starts = window_starts(end - begin)
                size = end - begin if len(starts) == 1 else WINDOW_TOKENS
                plans.append([(begin + start, begin + start + size) for start in starts])
            with ts.Batch():
                reads = [[store.data[a:b].read() for a, b in plan] for plan in plans]
            for index, futures in zip(batch, reads, strict=True):
                if remaining <= 0:
                    break
                begin, end = int(begins[index]), int(ends[index])
                windows = [
                    tokenizer.decode(f.result(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    for f in futures
                ]
                category = junk_verdict(windows).category
                audit.append({"shard": shard, "index": int(index), "tokens": end - begin, "category": category})
                if category is None:
                    kept.append(KeptDocument(shard, int(index), begin, end))
                    remaining -= end - begin
    if remaining > 0:
        logger.warning("%s: %d long tokens short of its %d budget", cell.name, remaining, cell.long_tokens)
    return kept, audit


def kept_records(cell: ReplayCell, kept: list[KeptDocument]) -> Iterator[dict[str, np.ndarray]]:
    stores: dict[str, JaggedArrayStore] = {}
    for document in kept:
        if document.shard not in stores:
            stores[document.shard] = JaggedArrayStore.open(
                str(StoragePath.parse(cell.source) / document.shard / "input_ids"), mode="r", dtype=np.int32
            )
        yield {"input_ids": stores[document.shard].data[document.start : document.stop].read().result()}


def build_cell(cell: ReplayCell, output: str, tokenizer_path: str, tokenizer) -> dict:
    """Write the cell's filtered long cache with its audit and return a summary row."""
    root = StoragePath.parse(output) / cell.name
    if (root / "summary.json").exists():
        return json.loads((root / "summary.json").read_text())
    kept, audit = select_kept_documents(cell, tokenizer)
    if kept:
        write_levanter_cache(
            kept_records(cell, kept),
            str(root / LONG_CACHE_NAME),
            metadata={"tokenizer": tokenizer_path, "length_threshold": LENGTH_THRESHOLD},
            batch_size=1,
            max_pending_batches=1,
        )
    (root / "verdicts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in audit))
    flagged = [row for row in audit if row["category"] is not None]
    summary = {
        "cell": cell.name,
        "budget_tokens": cell.long_tokens,
        "kept_documents": len(kept),
        "kept_tokens": sum(d.stop - d.start for d in kept),
        "judged_documents": len(audit),
        "flagged_documents": len(flagged),
        "flagged_tokens": sum(row["tokens"] for row in flagged),
        "shards_read": len({row["shard"] for row in audit}),
        "listed_shards": len(cell.shards),
    }
    (root / "summary.json").write_text(json.dumps(summary) + "\n")
    logger.info("%s", summary)
    return summary


@click.command()
@click.option("--cells", "cells_path", required=True, help="Gzipped replay preparation config of the reproduction.")
@click.option("--output", required=True, help="Prefix to receive <cell>/gt_64k, verdicts.jsonl and summary.json.")
@click.option("--tokenizer", "tokenizer_path", required=True, help="Tokenizer the source shards were built with.")
@click.option("--only", multiple=True, help="Build only these cells; repeat for several.")
@click.option("--workers", type=click.IntRange(min=1), default=8, show_default=True)
def main(cells_path: str, output: str, tokenizer_path: str, only: tuple[str, ...], workers: int) -> None:
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    tokenizer = load_tokenizer(tokenizer_path)
    cells = [cell for cell in read_cells(cells_path) if not only or cell.name in only]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        summaries = list(pool.map(lambda cell: build_cell(cell, output, tokenizer_path, tokenizer), cells))
    (StoragePath.parse(output) / "summary.jsonl").write_text("".join(json.dumps(row) + "\n" for row in summaries))


if __name__ == "__main__":
    main()
