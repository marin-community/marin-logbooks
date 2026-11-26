# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Version the competitive-programming SFT stores for mix V9 (V9 stages S2+S3).

User decisions 2026-11-22: build V9 = V8 plus (a) openthoughts4-code dropped,
(b) competitive-programming stores re-versioned with rows that duplicate RL task
prompts removed, and (c) the same stores re-versioned with every trace longer
than ~16k assistant tokens dropped. This module implements (b)+(c) for the two
competitive-programming sources of the V8 release; the old stores are never
overwritten.

A row is dropped when

- its assistant span (the ``loss_weights > 0`` mask) exceeds the length cap
  (stage ``assistant_length_cap``), or
- it duplicates an RL task prompt under the Q2 content-overlap rule
  (stage ``rl_lexical``): at least OVERLAP_MIN_MATCHED distinct 13-grams that
  the row shares with one RL prompt, covering at least OVERLAP_THRESHOLD of
  that prompt's non-generic features. Pool-common 13-grams — those carried by
  at least GENERIC_MIN_PROMPTS distinct RL prompts (the Q2 genericity rule over
  the full RL prompt population) — cannot count: the RL pool task specs share
  the generator's instruction preamble and stock competitive-programming
  phrasing with the store rows, and both the decontamination paragraph mark and
  pool-rare/store-common phrasing flagged 17-45% of rows in the earlier filter
  attempts.

The job runs in three passes: scan every row (bloom probe with the generic drop
set, candidates = rows with enough matched non-generic 13-grams), decide the
lexical drops in the driver against the per-prompt denominators, then rewrite
the kept rows into new content-addressed shard caches with a per-shard
``decisions.jsonl`` audit and receipts recording row/token deltas per stage.
The store-level transform receipt attributes every lexically-dropped row to the
RL prompt it duplicates.
"""

import argparse
import hashlib
import json
import logging
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import dupekit
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from levanter.store.cache import write_levanter_cache
from levanter.store.jagged_array import JaggedArrayStore
from marin.datakit.decon import NGramConfig, _bloom_hash, _extract_features, _load_bloom
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer

from experiments.sft_new_mix.exp3_runtime import configure_storage, read_pinned

logger = logging.getLogger(__name__)

LENGTH_CAP = 16_384
NGRAM = NGramConfig(ngram_length=13, stride=0, overlap_threshold=0.5, min_matched_features=8)
GENERIC_MIN_PROMPTS = 3
"""A 13-gram carried by at least this many RL prompts is boilerplate (the Q2 genericity rule,
applied over the full RL prompt population instead of the 380-prompt sweep sample)."""
OVERLAP_MIN_MATCHED = 8
"""Minimum distinct prompt-specific 13-grams a row must share with one RL prompt."""
OVERLAP_THRESHOLD = 0.2
"""Fraction of a prompt's non-generic features present in a row for a content match."""
BATCH_ROWS = 128
WRITE_BATCH = 128

# Worker-process state, set by _init_worker.
_BLOOM: dupekit.Bloom | None = None
_GENERIC_HASHES: frozenset[int] = frozenset()
_TOKENIZER: Tokenizer | None = None
_LENGTH_CAP = LENGTH_CAP


def shard_job(source: str, entry: dict, shard_key: str, output_root: str, length_cap: int) -> dict:
    """The pinned per-shard job spec; its JSON hash names the new shard directory."""
    ledger = entry["ledger"]
    return {
        "source": source,
        "base_cache_dir": entry["cache_dir"].rstrip("/"),
        "base_shard_key": shard_key,
        "base_rows": ledger["shard_rows"][shard_key],
        "base_tokens": ledger["field_counts_by_shard"][shard_key]["input_ids"],
        "loss_mode": ledger["metadata"]["preprocessor_metadata"]["loss_mode"],
        "length_cap": length_cap,
        "ngram": {
            "ngram_length": NGRAM.ngram_length,
            "stride": NGRAM.stride,
            "overlap_threshold": NGRAM.overlap_threshold,
            "min_matched_features": NGRAM.min_matched_features,
        },
        "generic_min_prompts": GENERIC_MIN_PROMPTS,
        "output_root": output_root.rstrip("/"),
    }


def store_dir(output_root: str, source: str) -> str:
    # The V8 store-name convention: the source's '/'-separated name with '__' joins,
    # e.g. sft/source/X -> sft__source__X (the name already starts with sft/).
    return f"{output_root.rstrip('/')}/{source.replace('/', '__')}/train"


def new_shard_uri(job: dict) -> str:
    identity = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    return f"{store_dir(job['output_root'], job['source'])}/shards/{identity}/cache"


def relative_shard_key(output_cache_uri: str, output_root: str, source: str) -> str:
    """The new shard's ledger key relative to the versioned store's cache_dir."""
    prefix = store_dir(output_root, source) + "/"
    if not output_cache_uri.startswith(prefix):
        raise ValueError(f"shard {output_cache_uri} does not live under its store dir {prefix}")
    return output_cache_uri[len(prefix) :]


def _init_worker(bloom_uri: str, tokenizer_json: str, length_cap: int, generic_hashes: frozenset[int]) -> None:
    global _BLOOM, _GENERIC_HASHES, _TOKENIZER, _LENGTH_CAP
    _BLOOM = _load_bloom(bloom_uri)
    _GENERIC_HASHES = generic_hashes
    _TOKENIZER = Tokenizer.from_str(tokenizer_json)
    _LENGTH_CAP = length_cap


def compute_generic_drop_set(index_parts: list[dict], work: Path) -> tuple[frozenset[int], dict]:
    """Hashes carried by >= GENERIC_MIN_PROMPTS distinct RL prompts are boilerplate."""
    part_hashes = []
    part_ids = []
    for part in index_parts:
        fs, key = filesystem_for(part["uri"])
        local = work / f"index-{Path(part['uri']).name}"
        local.write_bytes(fs.cat(key))
        table = pq.read_table(local, columns=["hash", "eval_id"])
        part_hashes.append(table.column("hash").combine_chunks())
        # large_string: the concatenated ids exceed the 32-bit string offset range.
        part_ids.append(table.column("eval_id").combine_chunks().cast(pa.large_string()))
        local.unlink()
    hash_col = pa.concat_arrays(part_hashes)
    # The multiplicity join runs on dictionary-encoded prompt ids (ints), not
    # strings: the string group-by OOMs at this index size.
    encoded = pa.compute.dictionary_encode(pa.concat_arrays(part_ids))
    if isinstance(encoded, pa.ChunkedArray):
        codes = pa.concat_arrays([chunk.indices for chunk in encoded.chunks]).cast(pa.int32())
    else:
        codes = encoded.indices.cast(pa.int32())
    pairs = pa.table({"hash": hash_col, "eid": codes})
    distinct_pairs = pairs.group_by(["hash", "eid"]).aggregate([("eid", "count")])
    per_hash = distinct_pairs.group_by("hash").aggregate([("eid", "count")])
    hashes = per_hash.column("hash").to_pylist()
    multiplicities = per_hash.column("eid_count").to_pylist()
    histogram: Counter[int] = Counter()
    generic: set[int] = set()
    for value, count in zip(hashes, multiplicities, strict=True):
        histogram[count] += 1
        if count >= GENERIC_MIN_PROMPTS:
            generic.add(value)
    evidence = {
        "generic_min_prompts": GENERIC_MIN_PROMPTS,
        "index_rows": int(sum(len(chunk) for chunk in part_hashes)),
        "distinct_hashes": len(hashes),
        "generic_hashes": len(generic),
        "multiplicity_histogram_head": {str(k): histogram[k] for k in sorted(histogram)[:12]},
    }
    return frozenset(generic), evidence


def load_index_matches(index_parts: list[dict], matched_hashes: set[int], work: Path) -> dict[int, list[str]]:
    """Join the matched hashes against the bloom's hash-index parts (hash -> eval ids)."""
    if not matched_hashes:
        return {}
    mapping: dict[int, list[str]] = {}
    for part in index_parts:
        fs, key = filesystem_for(part["uri"])
        local = work / f"index-{Path(part['uri']).name}"
        local.write_bytes(fs.cat(key))
        table = pq.read_table(local, columns=["hash", "eval_id"])
        hashes = table.column("hash").to_pylist()
        ids = table.column("eval_id").to_pylist()
        for value, eval_id in zip(hashes, ids, strict=True):
            if value in matched_hashes:
                mapping.setdefault(value, []).append(eval_id)
        local.unlink()
    return mapping


def count_denominator_part(job: tuple[str, str]) -> dict[str, int]:
    """Per RL prompt: the number of distinct non-generic (pool-rare) 13-gram features."""
    name, uri = job
    fs, key = filesystem_for(uri)
    blob = fs.cat(key)
    local = Path(tempfile.mkdtemp(prefix="v9-denom-")) / f"{name}.parquet"
    local.write_bytes(blob)
    table = pq.read_table(local, columns=["id", "text"])
    denominators = {}
    for eval_id, text in zip(table.column("id").to_pylist(), table.column("text").to_pylist(), strict=True):
        features = {h for f in _extract_features(text, NGRAM) if (h := _bloom_hash(f)) not in _GENERIC_HASHES}
        denominators[eval_id] = len(features)
    return denominators


def prompt_denominators(corpus: dict, generic: frozenset[int], workers: int) -> dict[str, int]:
    """Per RL prompt: the number of distinct non-generic (pool-rare) 13-gram features."""
    global _GENERIC_HASHES
    jobs = [(name, stats["uri"]) for name, stats in sorted(corpus.items())]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_denominator_worker, initargs=(generic,)) as pool:
        parts = list(pool.map(count_denominator_part, jobs))
    merged: dict[str, int] = {}
    for part in parts:
        merged.update(part)
    return merged


def _init_denominator_worker(generic: frozenset[int]) -> None:
    global _GENERIC_HASHES
    _GENERIC_HASHES = generic


def scan_shard(job: dict) -> dict:
    """Pass 1: length drops + lexical candidates (rows with enough matched non-generic 13-grams)."""
    if _BLOOM is None or _TOKENIZER is None:
        raise RuntimeError("worker state missing: _init_worker has not run")
    source = job["source"]
    part = f"{job['base_cache_dir'].rstrip('/')}/{job['base_shard_key']}"
    ids_store = JaggedArrayStore.open(f"{part}/input_ids", mode="r", dtype=np.int32)
    weights_store = JaggedArrayStore.open(f"{part}/loss_weights", mode="r", dtype=np.float32)
    if len(ids_store) != job["base_rows"] or len(weights_store) != job["base_rows"]:
        raise ValueError(f"{source} {job['base_shard_key']}: base shard row count changed")

    length_dropped: list[dict] = []
    kept_after_length: list[int] = []
    candidates: list[dict] = []
    drop_counts = {"assistant_length_cap": {"rows": 0, "tokens": 0}}
    for start in range(0, job["base_rows"], BATCH_ROWS):
        indices = range(start, min(start + BATCH_ROWS, job["base_rows"]))
        id_rows = ids_store.get_batch_sync(indices)
        weight_rows = weights_store.get_batch_sync(indices)
        for index, ids, weights in zip(indices, id_rows, weight_rows, strict=True):
            if len(ids) != len(weights):
                raise ValueError(f"{source} row {index}: unequal field lengths")
            assistant_tokens = int((weights > 0).sum())
            row_tokens = len(ids)
            if assistant_tokens > _LENGTH_CAP:
                drop_counts["assistant_length_cap"]["rows"] += 1
                drop_counts["assistant_length_cap"]["tokens"] += row_tokens
                length_dropped.append(
                    {
                        "row": index,
                        "stage": "assistant_length_cap",
                        "tokens": row_tokens,
                        "reason": f"{assistant_tokens} assistant tokens > cap {_LENGTH_CAP}",
                    }
                )
                continue
            kept_after_length.append(index)
            text = _TOKENIZER.decode(ids[weights == 0].tolist(), skip_special_tokens=False)
            if not text:
                continue
            matched = set()
            for feature in _extract_features(text, NGRAM):
                value = _bloom_hash(feature)
                if value in _BLOOM and value not in _GENERIC_HASHES:
                    matched.add(value)
            if len(matched) >= OVERLAP_MIN_MATCHED:
                candidates.append({"row": index, "tokens": row_tokens, "hashes": sorted(matched)})

    return {
        "shard_key": job["base_shard_key"],
        "rows": job["base_rows"],
        "length_dropped": length_dropped,
        "length_drop_counts": drop_counts,
        "kept_after_length": kept_after_length,
        "candidates": candidates,
    }


def write_shard(job: dict, kept_indices: list[int], length_dropped: list[dict], lexical_dropped: list[dict]) -> dict:
    """Pass 2: rewrite the kept rows into the new versioned shard cache with a full audit."""
    source = job["source"]
    part = f"{job['base_cache_dir'].rstrip('/')}/{job['base_shard_key']}"
    ids_store = JaggedArrayStore.open(f"{part}/input_ids", mode="r", dtype=np.int32)
    weights_store = JaggedArrayStore.open(f"{part}/loss_weights", mode="r", dtype=np.float32)
    if len(ids_store) != job["base_rows"] or len(weights_store) != job["base_rows"]:
        raise ValueError(f"{source} {job['base_shard_key']}: base shard row count changed")

    dropped = sorted(length_dropped + lexical_dropped, key=lambda record: record["row"])
    lengths: list[int] = []
    supervised = 0
    for start in range(0, len(kept_indices), BATCH_ROWS):
        batch = kept_indices[start : start + BATCH_ROWS]
        weight_rows = weights_store.get_batch_sync(batch)
        for weights in weight_rows:
            supervised += int((weights > 0).sum())
        id_rows = ids_store.get_batch_sync(batch)
        for ids in id_rows:
            lengths.append(len(ids))

    output_uri = new_shard_uri(job)
    expected_digest = hashlib.sha256()

    def records():
        for start in range(0, len(kept_indices), WRITE_BATCH):
            batch = kept_indices[start : start + WRITE_BATCH]
            id_rows = ids_store.get_batch_sync(batch)
            weight_rows = weights_store.get_batch_sync(batch)
            for ids, weights in zip(id_rows, weight_rows, strict=True):
                expected_digest.update(ids.astype("<i4", copy=False).tobytes())
                expected_digest.update(weights.astype("<f4", copy=False).tobytes())
                yield {"input_ids": ids, "loss_weights": weights}

    result = write_levanter_cache(
        records(),
        output_uri,
        metadata={
            "source": source,
            "loss_mode": job["loss_mode"],
            "loss_weights_indexing": "prediction",
            "base_shard_key": job["base_shard_key"],
            "v9_filter_job_sha256": hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
        },
        batch_size=WRITE_BATCH,
        max_pending_batches=2,
    )
    if result["count"] != len(kept_indices):
        raise ValueError(f"{source} {job['base_shard_key']}: output row count differs from the selection")

    shard_root = output_uri.rsplit("/cache", 1)[0]
    decisions = "\n".join(json.dumps(record, sort_keys=True) for record in dropped) + ("\n" if dropped else "")
    StoragePath(f"{shard_root}/decisions.jsonl").write_text(decisions)
    receipt = {
        "job": job,
        "complete": True,
        "source": source,
        "output_cache_uri": output_uri,
        "input_rows": job["base_rows"],
        "input_tokens": job["base_tokens"],
        "output_rows": len(kept_indices),
        "output_tokens": sum(lengths),
        "supervised_prediction_positions": supervised,
        "lengths": lengths,
        "dropped_rows": dropped,
        "output_tokens_and_weights_sha256": expected_digest.hexdigest(),
    }
    StoragePath(f"{shard_root}/receipt.json").write_text(json.dumps(receipt, sort_keys=True))
    return {
        "shard_key": job["base_shard_key"],
        "new_shard_key": relative_shard_key(output_uri, job["output_root"], source),
        "receipt_uri": f"{shard_root}/receipt.json",
        "output_rows": len(kept_indices),
        "output_tokens": sum(lengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-uri", required=True, help="The V8 cache catalog URI")
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--source", action="append", required=True, help="A comp-prog source name; repeatable")
    parser.add_argument("--bloom-manifest-uri", required=True, help="The S1 rl-prompt bloom manifest")
    parser.add_argument("--bloom-manifest-sha256", required=True)
    parser.add_argument("--tokenizer-uri", required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--length-cap", type=int, default=LENGTH_CAP)
    parser.add_argument("--output-root", required=True, help="The V9 versioned-store root")
    parser.add_argument("--destination", required=True, help="S3 prefix for the transform receipt")
    parser.add_argument("--work", default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()

    catalog = json.loads(read_pinned(args.catalog_uri, args.catalog_sha256).decode())["splits"]["train"]
    bloom_manifest = json.loads(read_pinned(args.bloom_manifest_uri, args.bloom_manifest_sha256).decode())
    bloom_uri = bloom_manifest["bloom"]["uri"]
    tokenizer = Tokenizer.from_str(read_pinned(args.tokenizer_uri, args.tokenizer_sha256).decode())

    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="v9-comp-prog-"))
    work.mkdir(parents=True, exist_ok=True)

    logger.info(
        "computing the generic (pool-common) 13-gram drop set from %d index parts", len(bloom_manifest["index_parts"])
    )
    generic, generic_evidence = compute_generic_drop_set(bloom_manifest["index_parts"], work)
    logger.info(
        "generic drop set: %d hashes (of %d distinct)",
        generic_evidence["generic_hashes"],
        generic_evidence["distinct_hashes"],
    )
    denominators = prompt_denominators(bloom_manifest["sources"], generic, args.workers)
    logger.info("prompt denominators: %d prompts", len(denominators))

    receipt = {
        "status": "COMPLETE",
        "stage": "V9 S2+S3 competitive-programming store versioning",
        "pins": {
            "catalog_uri": args.catalog_uri,
            "catalog_sha256": args.catalog_sha256,
            "bloom_manifest_uri": args.bloom_manifest_uri,
            "bloom_manifest_sha256": args.bloom_manifest_sha256,
            "tokenizer_uri": args.tokenizer_uri,
            "tokenizer_sha256": args.tokenizer_sha256,
            "length_cap": args.length_cap,
            "ngram": {
                "ngram_length": NGRAM.ngram_length,
                "stride": NGRAM.stride,
                "overlap_threshold": NGRAM.overlap_threshold,
                "min_matched_features": NGRAM.min_matched_features,
            },
            "generic_min_prompts": GENERIC_MIN_PROMPTS,
            "overlap_min_matched": OVERLAP_MIN_MATCHED,
            "overlap_threshold": OVERLAP_THRESHOLD,
            "output_root": args.output_root,
        },
        "generic_drop_set": generic_evidence,
        "sources": {},
    }

    for source in args.source:
        entry = catalog[source]
        shards = sorted(entry["ledger"]["field_counts_by_shard"])
        jobs = [shard_job(source, entry, shard_key, args.output_root, args.length_cap) for shard_key in shards]
        logger.info("%s: scanning %d shards", source, len(jobs))
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(bloom_uri, tokenizer.to_str(), args.length_cap, generic),
        ) as pool:
            scans = list(pool.map(scan_shard, jobs))

        # Decide the lexical drops under the Q2 content-overlap rule.
        candidate_hashes: set[int] = set()
        for scan in scans:
            for candidate in scan["candidates"]:
                candidate_hashes.update(candidate["hashes"])
        index = load_index_matches(bloom_manifest["index_parts"], candidate_hashes, work)
        lexical_by_shard: dict[str, list[dict]] = {scan["shard_key"]: [] for scan in scans}
        candidate_tokens = {
            (scan["shard_key"], candidate["row"]): candidate["tokens"]
            for scan in scans
            for candidate in scan["candidates"]
        }
        attribution: dict[str, int] = {}
        attribution_examples: list[dict] = []
        for scan in scans:
            for candidate in scan["candidates"]:
                per_prompt: Counter[str] = Counter()
                for value in candidate["hashes"]:
                    for eval_id in index.get(value, []):
                        per_prompt[eval_id] += 1
                for eval_id, numerator in per_prompt.items():
                    denominator = denominators.get(eval_id, 0)
                    if numerator >= OVERLAP_MIN_MATCHED and denominator and numerator >= OVERLAP_THRESHOLD * denominator:
                        lexical_by_shard[scan["shard_key"]].append(
                            {
                                "row": candidate["row"],
                                "stage": "rl_lexical",
                                "tokens": candidate_tokens[(scan["shard_key"], candidate["row"])],
                                "reason": (
                                    f"duplicates RL prompt {eval_id}: {numerator} distinct matched 13-grams = "
                                    f"{numerator / denominator:.0%} of its {denominator} non-generic features"
                                ),
                                "prompt": eval_id,
                                "matched_features": numerator,
                                "denominator": denominator,
                            }
                        )
                        attribution[eval_id] = attribution.get(eval_id, 0) + 1
                        if len(attribution_examples) < 40:
                            attribution_examples.append(
                                {
                                    "row": candidate["row"],
                                    "prompt": eval_id,
                                    "matched": numerator,
                                    "denominator": denominator,
                                    "shard": scan["shard_key"],
                                }
                            )
                        break

        logger.info("%s: rewriting %d shards", source, len(jobs))
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(bloom_uri, tokenizer.to_str(), args.length_cap, generic),
        ) as pool:
            writes = []
            for scan in scans:
                lexical = lexical_by_shard[scan["shard_key"]]
                lexical_rows = {record["row"] for record in lexical}
                kept = [row for row in scan["kept_after_length"] if row not in lexical_rows]
                writes.append((scan, kept))
            results = list(
                pool.map(
                    write_shard,
                    jobs,
                    [kept for _, kept in writes],
                    [scan["length_dropped"] for scan, _ in writes],
                    [lexical_by_shard[scan["shard_key"]] for scan, _ in writes],
                )
            )

        output_tokens = sum(result["output_tokens"] for result in results)
        output_rows = sum(result["output_rows"] for result in results)
        length_drops = {
            "rows": sum(scan["length_drop_counts"]["assistant_length_cap"]["rows"] for scan in scans),
            "tokens": sum(scan["length_drop_counts"]["assistant_length_cap"]["tokens"] for scan in scans),
        }
        lexical_drops = {
            "rows": sum(len(records) for records in lexical_by_shard.values()),
            "tokens": sum(record["tokens"] for records in lexical_by_shard.values() for record in records),
        }
        base_tokens = entry["ledger"]["field_counts"]["input_ids"]
        store_base_rows = entry["ledger"]["total_num_rows"]
        if length_drops["rows"] + lexical_drops["rows"] + output_rows != store_base_rows:
            raise ValueError(f"{source}: row population does not conserve")
        if length_drops["tokens"] + lexical_drops["tokens"] + output_tokens != base_tokens:
            raise ValueError(f"{source}: token population does not conserve")
        store_receipt = {
            "source": source,
            "base": {
                "cache_dir": entry["cache_dir"],
                "rows": store_base_rows,
                "tokens": base_tokens,
                "shards": len(shards),
            },
            "output": {
                "cache_dir": store_dir(args.output_root, source),
                "rows": output_rows,
                "tokens": output_tokens,
                "shards": len(results),
                "row_delta": output_rows - entry["ledger"]["total_num_rows"],
                "token_delta": output_tokens - base_tokens,
            },
            "drops": {
                "assistant_length_cap": length_drops,
                "rl_lexical": lexical_drops,
            },
            "shards": results,
            "rl_prompt_attribution_rows": dict(sorted(attribution.items(), key=lambda kv: -kv[1])),
            "rl_prompt_attribution_examples": attribution_examples,
        }
        receipt["sources"][source] = store_receipt
        logger.info(
            "%s: rows %d -> %d, tokens %d -> %d, length drops %d, lexical drops %d",
            source,
            store_receipt["base"]["rows"],
            output_rows,
            base_tokens,
            output_tokens,
            length_drops["rows"],
            lexical_drops["rows"],
        )

    document = json.dumps(receipt, sort_keys=True, indent=1).encode()
    fs, key = filesystem_for(f"{args.destination.rstrip('/')}/transform-receipt.json")
    fs.pipe(key, document)
    print(
        json.dumps(
            {
                "generic": receipt["generic_drop_set"],
                "sources": {
                    source: {
                        "rows": f'{data["base"]["rows"]}->{data["output"]["rows"]}',
                        "tokens": f'{data["base"]["tokens"]}->{data["output"]["tokens"]}',
                        "drops": data["drops"],
                        "rl_prompts_matched": len(data["rl_prompt_attribution_rows"]),
                    }
                    for source, data in receipt["sources"].items()
                },
            },
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
