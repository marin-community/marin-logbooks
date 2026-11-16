# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a held-out long replay set labelled by the junk filter, one document per row.

Each validation cell of the reproduction's replay preparation names one held-out source shard that no
train cell lists; the filtered builder never reached them either. From each, this takes the first
``documents_per_cell`` long documents in index order, judges each exactly as ``filtered_replay`` does,
and keeps at most its first ``tokens`` tokens. Rows are stored as processed chat records whose
``assistant_masks`` cover every token, so an unpacked chat dataset scores each document alone, padded
to the eval length, on every position that predicts one of its tokens. The manifest lists the rows in
cache order with each verdict and the number of positions the eval will score.
"""

import gzip
import json
import logging
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

from experiments.sft_cold_start.filtered_replay import WINDOW_TOKENS, window_starts
from experiments.sft_cold_start.replay_subset import LENGTH_THRESHOLD

logger = logging.getLogger(__name__)

VALIDATION_SUFFIX = "-validation"
MANIFEST_NAME = "manifest.jsonl"


@dataclass(frozen=True)
class HeldOutDocument:
    cell: str
    shard: str
    index: int
    tokens: int
    category: str | None
    input_ids: np.ndarray


def read_validation_cells(path: str) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        config = yaml.safe_load(handle)
    cells = [cell for cell in config["cells"] if cell["name"].endswith(VALIDATION_SUFFIX)]
    for cell in cells:
        if len(cell["shards"]) != 1:
            raise ValueError(f"{cell['name']} lists {len(cell['shards'])} shards, expected one held-out shard")
    return sorted(cells, key=lambda cell: cell["name"])


def cell_documents(cell: dict, tokenizer, *, documents_per_cell: int, tokens: int) -> list[HeldOutDocument]:
    """The cell's first long documents, judged and truncated."""
    shard = cell["shards"][0]
    store = JaggedArrayStore.open(str(StoragePath.parse(cell["source"]) / shard / "input_ids"), mode="r", dtype=np.int32)
    count = len(store)
    offsets = np.asarray(store.offsets[: count + 1].read().result())
    ends = offsets[1:].astype(np.int64)
    begins = np.concatenate((np.array([0], dtype=np.int64), ends[:-1]))
    chosen = np.flatnonzero(ends - begins > LENGTH_THRESHOLD)[:documents_per_cell]
    plans = []
    for index in chosen:
        begin, end = int(begins[index]), int(ends[index])
        plans.append([(begin + start, begin + start + WINDOW_TOKENS) for start in window_starts(end - begin)])
    with ts.Batch():
        window_reads = [[store.data[a:b].read() for a, b in plan] for plan in plans]
        prefix_reads = [store.data[int(begins[i]) : min(int(ends[i]), int(begins[i]) + tokens)].read() for i in chosen]
    documents = []
    for index, windows, prefix in zip(chosen, window_reads, prefix_reads, strict=True):
        text = [
            tokenizer.decode(w.result(), skip_special_tokens=False, clean_up_tokenization_spaces=False) for w in windows
        ]
        documents.append(
            HeldOutDocument(
                cell=cell["name"],
                shard=shard,
                index=int(index),
                tokens=int(ends[index] - begins[index]),
                category=junk_verdict(text).category,
                input_ids=np.asarray(prefix.result(), dtype=np.int32),
            )
        )
    logger.info(
        "%s: %d documents, %d flagged", cell["name"], len(documents), sum(d.category is not None for d in documents)
    )
    return documents


@click.command()
@click.option("--cells", "cells_path", required=True, help="Gzipped replay preparation config of the reproduction.")
@click.option("--output", required=True, help="Prefix to receive validation/ (the cache) and manifest.jsonl.")
@click.option("--tokenizer", "tokenizer_path", required=True, help="Tokenizer the source shards were built with.")
@click.option("--documents-per-cell", type=click.IntRange(min=1), required=True)
@click.option("--tokens", type=click.IntRange(min=2), required=True, help="Most tokens kept per row; the eval length.")
@click.option("--workers", type=click.IntRange(min=1), default=8, show_default=True)
def main(cells_path: str, output: str, tokenizer_path: str, documents_per_cell: int, tokens: int, workers: int) -> None:
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    root = StoragePath.parse(output)
    if (root / MANIFEST_NAME).exists():
        raise FileExistsError(f"{root} already holds a held-out set")
    tokenizer = load_tokenizer(tokenizer_path)
    cells = read_validation_cells(cells_path)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_cell = list(
            pool.map(
                lambda cell: cell_documents(cell, tokenizer, documents_per_cell=documents_per_cell, tokens=tokens), cells
            )
        )
    documents = [document for group in per_cell for document in group]
    write_levanter_cache(
        (
            {"input_ids": document.input_ids, "assistant_masks": np.ones_like(document.input_ids)}
            for document in documents
        ),
        str(root / "validation"),
        metadata={"tokenizer": tokenizer_path, "tokens_per_row": tokens},
        batch_size=1,
        max_pending_batches=1,
    )
    (root / MANIFEST_NAME).write_text(
        "".join(
            json.dumps(
                {
                    "cell": d.cell,
                    "shard": d.shard,
                    "index": d.index,
                    "tokens": d.tokens,
                    "category": d.category,
                    "scored_tokens": len(d.input_ids) - 1,
                }
            )
            + "\n"
            for d in documents
        )
    )
    flagged = sum(d.category is not None for d in documents)
    logger.info("Wrote %d documents from %d cells, %d flagged, to %s", len(documents), len(cells), flagged, root)


if __name__ == "__main__":
    main()
