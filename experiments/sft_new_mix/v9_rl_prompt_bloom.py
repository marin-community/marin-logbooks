# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the V9 RL-prompt lexical corpus and its decontamination bloom (V9 stage S1).

Mix V9 removes competitive-programming SFT rows that duplicate RL task prompts.
The drop rule is the standard lexical-decontamination mark (word 13-grams,
``marin.datakit.decon``) against the RL task prompt population the long loop
trains on:

- the nonagentic RL pool (`rl-pool-nonagentic/2026.11.18.71/train.parquet`),
- the agentic release (`swe-verified-20261112-v4/release/agentic-train.parquet`),
- its validation split (`validation.parquet`).

Each parquet row carries a ``task_spec`` JSON document with ``id`` and
``context.events`` messages. This job parses every task spec, extracts the user
messages as the prompt text, uploads a normalized (id, text) corpus, and builds
the ``dupekit`` bloom plus per-worker hash-index parts (decon's
``{hash, eval_id}`` schema) that the comp-prog filter scans against. Input
SHA-256 digests are computed while streaming and recorded in the manifest, so
downstream stages pin the manifest instead of each raw input.
"""

import argparse
import hashlib
import json
import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import dupekit
import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.decon import NGramConfig, _bloom_hash, _extract_features
from rigging.filesystem.buckets import filesystem_for

from experiments.sft_new_mix.exp3_runtime import configure_storage

logger = logging.getLogger(__name__)

NGRAM = NGramConfig(ngram_length=13, stride=0, overlap_threshold=0.5, min_matched_features=2)
BLOOM_FALSE_POSITIVE_RATE = 1e-9
BLOOM_SIZE_HEADROOM = 1.05
WORKERS = 16
BATCH_SIZE = 512

SOURCES = {
    "nonagentic-train": (
        "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp4/long-loop-exp4/"
        "rl-pool-nonagentic/2026.11.18.71/train.parquet"
    ),
    "agentic-train": (
        "s3://marin-us-east-08a/marin/users/benfeuer/long-loop/rl-data/"
        "swe-verified-20261112-v4/release/agentic-train.parquet"
    ),
    "validation": (
        "s3://marin-us-east-08a/marin/users/benfeuer/long-loop/rl-data/"
        "swe-verified-20261112-v4/release/validation.parquet"
    ),
}


def user_text_from_task_spec(task_spec: str, source: str, row_index: int) -> tuple[str, str]:
    """Parse one task_spec JSON into (eval_id, user-prompt text)."""
    try:
        spec = json.loads(task_spec)
    except json.JSONDecodeError as error:
        raise ValueError(f"{source} row {row_index}: task_spec is not JSON ({error})") from error
    eval_id = str(spec.get("id") or f"{source}::{row_index}")
    events = (spec.get("context") or {}).get("events") or []
    contents = [event.get("content") or "" for event in events if event.get("role") == "user"]
    text = "\n\n".join(part for part in contents if part)
    if not text:
        raise ValueError(f"{source} row {row_index} ({eval_id}): task_spec has no user message")
    return eval_id, text


def stream_sha256(uri: str) -> str:
    fs, key = filesystem_for(uri)
    digest = hashlib.sha256()
    with fs.open(key, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source(source: str, uri: str) -> tuple[list[str], list[str], dict]:
    """Parse one parquet source into (ids, texts) plus its digests."""
    fs, key = filesystem_for(uri)
    rows: list[tuple[str, str]] = []
    with fs.open(key, "rb") as stream:
        reader = pq.ParquetFile(stream)
        for chunk in reader.iter_batches(columns=["task_spec"], batch_size=BATCH_SIZE):
            values = chunk.column("task_spec").to_pylist()
            for offset, value in enumerate(values):
                rows.append(user_text_from_task_spec(value, source, len(rows) + offset))
    ids = [eval_id for eval_id, _ in rows]
    texts = [text for _, text in rows]
    content_digest = hashlib.sha256()
    for eval_id, text in rows:
        content_digest.update(f"{eval_id}\x00{text}\x00".encode())
    return (
        ids,
        texts,
        {
            "uri": uri,
            "rows": len(rows),
            "input_sha256": stream_sha256(uri),
            "parsed_sha256": content_digest.hexdigest(),
        },
    )


def upload_corpus(corpus_prefix: str, source: str, ids: list[str], texts: list[str]) -> dict:
    table = pa.table({"id": ids, "text": texts})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    blob = sink.getvalue().to_pybytes()
    uri = f"{corpus_prefix}/{source}.parquet"
    fs, key = filesystem_for(uri)
    fs.pipe(key, blob)
    return {"uri": uri, "rows": len(ids), "sha256": hashlib.sha256(blob).hexdigest()}


def build_part(job: tuple) -> dict:
    """Worker: extract features for one corpus chunk; write a partial bloom + index part."""
    part_path, rows, expected_items, part_index = job
    part = Path(part_path)
    part.mkdir(parents=True, exist_ok=True)
    bloom = dupekit.Bloom(expected_items, BLOOM_FALSE_POSITIVE_RATE)
    hashes: list[int] = []
    eval_ids: list[str] = []
    features = 0
    for eval_id, text in rows:
        seen: set[int] = set()
        for feature in _extract_features(text, NGRAM):
            value = _bloom_hash(feature)
            features += 1
            if value not in seen:
                seen.add(value)
                hashes.append(value & 0xFFFFFFFFFFFFFFFF)
                eval_ids.append(eval_id)
            bloom.add(value)
    table = pa.table(
        {"hash": pa.array(hashes, type=pa.uint64()), "eval_id": eval_ids},
        schema=pa.schema({"hash": pa.uint64(), "eval_id": pa.string()}),
    )
    index_file = part / "index.parquet"
    pq.write_table(table, index_file)
    bloom_file = part / "bloom.bin"
    bloom.save(str(bloom_file))
    return {
        "part": part_index,
        "rows": len(rows),
        "features": features,
        "index_rows": len(hashes),
        "bloom": str(bloom_file),
        "index": str(index_file),
    }


def merge_blooms(partial_paths: list[str], output_uri: str) -> dict:
    merged = dupekit.Bloom.load(partial_paths[0])
    for path in partial_paths[1:]:
        merged.update(dupekit.Bloom.load(path))
    blob = merged.save_bytes()
    uri = f"{output_uri}/filter.bin"
    fs, key = filesystem_for(uri)
    fs.pipe(key, blob)
    return {"uri": uri, "sha256": hashlib.sha256(blob).hexdigest(), "approx_items": float(merged.approx_items)}


def upload(blob: bytes, uri: str) -> dict:
    fs, key = filesystem_for(uri)
    fs.pipe(key, blob)
    return {"uri": uri, "sha256": hashlib.sha256(blob).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", required=True, help="Destination prefix for corpus + bloom")
    parser.add_argument("--work", default=None, help="Local work directory (default: a fresh temp dir)")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()

    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="v9-rl-bloom-"))
    work.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix.rstrip("/")

    corpus = {}
    prompts: list[tuple[str, str]] = []
    for source, uri in sorted(SOURCES.items()):
        ids, texts, stats = read_source(source, uri)
        corpus[source] = stats | upload_corpus(f"{prefix}/corpus", source, ids, texts)
        prompts.extend(zip(ids, texts, strict=True))
        logger.info("%s: %d prompts", source, len(ids))
    total_tokens = sum(len(text.split()) for _, text in prompts)
    logger.info("corpus: %d prompts, %d prompt tokens", len(prompts), total_tokens)

    expected_items = max(16, int(total_tokens * BLOOM_SIZE_HEADROOM))
    chunk_size = max(1, -(-len(prompts) // args.workers))
    jobs = []
    for part_index in range(args.workers):
        chunk = prompts[part_index * chunk_size : (part_index + 1) * chunk_size]
        if not chunk:
            continue
        jobs.append((str(work / f"part-{part_index:05d}"), chunk, expected_items, part_index))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        parts = list(pool.map(build_part, jobs))

    bloom = merge_blooms([part["bloom"] for part in parts], f"{prefix}/_bloom")
    fs, _ = filesystem_for(prefix + "/")
    index_parts = []
    for part in parts:
        blob = Path(part["index"]).read_bytes()
        index_parts.append(
            upload(blob, f"{prefix}/_bloom/index/part-{part['part']:05d}.parquet") | {"rows": part["index_rows"]}
        )

    manifest = {
        "status": "COMPLETE",
        "stage": "V9 S1 rl-prompt corpus + bloom",
        "config": {
            "ngram": {
                "ngram_length": NGRAM.ngram_length,
                "stride": NGRAM.stride,
                "overlap_threshold": NGRAM.overlap_threshold,
                "min_matched_features": NGRAM.min_matched_features,
            },
            "bloom_false_positive_rate": BLOOM_FALSE_POSITIVE_RATE,
            "expected_items": expected_items,
            "workers": args.workers,
        },
        "sources": corpus,
        "totals": {
            "prompts": len(prompts),
            "prompt_tokens": total_tokens,
            "features": sum(part["features"] for part in parts),
            "index_rows": sum(part["index_rows"] for part in parts),
        },
        "bloom": bloom,
        "index_parts": index_parts,
        "parts": [{k: v for k, v in part.items() if k not in ("bloom", "index")} for part in parts],
    }
    document = json.dumps(manifest, sort_keys=True, indent=1).encode()
    fs, key = filesystem_for(f"{prefix}/manifest.json")
    fs.pipe(key, document)
    print(json.dumps({k: manifest[k] for k in ("totals", "bloom", "config")}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
