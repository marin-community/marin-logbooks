# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Draw a share-weighted sample of replay documents as decoded token windows.

Scratch sampler for developing the replay junk filter. For each replay cache of
the 10.05 SFT mix it picks documents at random and, for each one, decodes eight
2,048-token windows: the head, the tail and six stratified-random interior
positions. Documents of at most 16,384 tokens are decoded whole. Only the
windows are read, so a row of hundreds of millions of tokens costs eight small
reads. Run it in us-central2, next to the caches.
"""

import asyncio
import gzip
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.store.cache import TreeCache
from levanter.utils.thread_utils import blocking_wait
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

REPLAY_PATH = Path(__file__).with_name("replay_skew8.json")
TOKENIZER = "gs://marin-us-central2/grug_sft/tokenizer/2026.09.18"
REPLAY_PATH_PATTERN = re.compile(r"/cluster=(\d+)/quality=(\d+)/length=(\w+)$")
WINDOW_TOKENS = 2_048
INTERIOR_WINDOWS = 6
WHOLE_DOCUMENT_TOKENS = WINDOW_TOKENS * (INTERIOR_WINDOWS + 2)
# Documents drawn per length stratum, split across caches by share with a floor.
STRATUM_DOCUMENTS = {"gt_64k": 12_000, "lte_64k": 4_000}
FLOOR_PER_CACHE = 8
READ_WORKERS = 32
CONCURRENT_READS = 64
SHARD_DOCUMENTS = 500
FIELD = "input_ids"
EXEMPLAR = {FIELD: np.zeros((0,), dtype=np.int32)}


def replay_caches() -> list[dict]:
    caches = []
    for _, record in sorted(json.loads(REPLAY_PATH.read_text()).items()):
        total_copies = sum(record["copies"].values())
        for path, count in sorted(record["copies"].items()):
            match = REPLAY_PATH_PATTERN.search(path)
            if match is None:
                raise ValueError(f"Unrecognized replay cache path: {path}")
            cluster, quality, length = match.groups()
            caches.append(
                {
                    "source": f"replay/c{int(cluster):02d}/q{quality}/{length}",
                    "stratum": length,
                    "share_of_replay": record["weight"] * count / total_copies,
                    "path": path,
                }
            )
    return caches


def window_starts(length: int, rng: np.random.Generator) -> list[int]:
    """Head, six interior positions (one per equal stratum of the interior) and tail."""
    if length <= WHOLE_DOCUMENT_TOKENS:
        return [0]
    interior_start, interior_stop = WINDOW_TOKENS, length - 2 * WINDOW_TOKENS
    edges = np.linspace(interior_start, interior_stop, INTERIOR_WINDOWS + 1).astype(np.int64)
    interior = [int(rng.integers(low, max(low + 1, high))) for low, high in zip(edges[:-1], edges[1:], strict=True)]
    return [0, *interior, length - WINDOW_TOKENS]


async def read_documents(cache: TreeCache, indices: list[int], rng: np.random.Generator) -> list[tuple]:
    """Read each document's length and its sampled token windows, never the whole row of a long document."""
    limit = asyncio.Semaphore(CONCURRENT_READS)

    async def read_document(index: int, starts_seed: int) -> tuple:
        async with limit:
            begin, end = (int(v) for v in await cache.flat_field_document_offsets(FIELD, index, index + 1))
            length = end - begin
            starts = window_starts(length, np.random.default_rng(starts_seed))
            size = length if len(starts) == 1 else WINDOW_TOKENS
            windows = await cache.get_flat_field_batch(FIELD, [begin + offset for offset in starts], size)
            return index, length, starts, windows

    seeds = rng.integers(0, 2**62, size=len(indices))
    return await asyncio.gather(*(read_document(i, int(s)) for i, s in zip(indices, seeds, strict=True)))


def sample_cache(cache_info: dict, wanted: int, seed: list[int], tokenizer) -> list[dict]:
    rng = np.random.default_rng(seed)
    cache = TreeCache.load(cache_info["path"], EXEMPLAR)
    documents = len(cache)
    indices = sorted(int(i) for i in rng.choice(documents, size=min(wanted, documents), replace=False))
    rows = []
    for index, length, starts, windows in blocking_wait(read_documents(cache, indices, rng)):
        rows.append(
            {
                "source": cache_info["source"],
                "share_of_replay": cache_info["share_of_replay"],
                "documents_in_source": documents,
                "doc_index": index,
                "tokens": length,
                "windows": [
                    {
                        "start": offset,
                        "text": tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                    }
                    for offset, ids in zip(starts, windows, strict=True)
                ],
            }
        )
    logger.info("%s: %d of %d documents", cache_info["source"], len(rows), documents)
    return rows


def main(output: str, seed: int) -> None:
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    tokenizer = load_tokenizer(TOKENIZER)
    caches = replay_caches()
    jobs = []
    for stratum, budget in STRATUM_DOCUMENTS.items():
        members = [(number, cache) for number, cache in enumerate(caches) if cache["stratum"] == stratum]
        total_share = sum(cache["share_of_replay"] for _, cache in members)
        for number, cache in members:
            wanted = max(FLOOR_PER_CACHE, round(budget * cache["share_of_replay"] / total_share))
            jobs.append((cache, wanted, [seed, number]))
    with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
        sampled = list(pool.map(lambda job: sample_cache(*job, tokenizer), jobs))
    base = StoragePath.parse(output)
    for stratum in STRATUM_DOCUMENTS:
        rows = [row for (cache, _, _), group in zip(jobs, sampled, strict=True) if cache["stratum"] == stratum for row in group]
        for shard, begin in enumerate(range(0, len(rows), SHARD_DOCUMENTS)):
            lines = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows[begin : begin + SHARD_DOCUMENTS])
            (base / f"{stratum}-{shard:03d}.jsonl.gz").write_bytes(gzip.compress(lines.encode("utf-8", "surrogatepass")))
        logger.info("%s: wrote %d documents", stratum, len(rows))
    (base / "DONE").write_text(json.dumps({"seed": seed, "documents": sum(len(group) for group in sampled)}))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
