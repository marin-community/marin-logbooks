# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""V10.1 stage A: apply the extracted tool-conversion deltas to the three math
sources' V10 stores.

Only the delta-bearing shards are rewritten. For each replaced row, the
payload ``input_ids`` are substituted at their mapped ``output_row`` and the
row's ``loss_weights`` are recomputed with the source's own supervision mode
(``chat_assistant_only`` — V10's supervision is kept; the all-position loss for
Exp7 is applied by the LLOM trainer at consumption and the cached masks are
ignored). Unchanged rows keep V10's ``input_ids`` and ``loss_weights`` bytes,
and shards with no deltas are not rewritten at all: the receipt points them at
their unchanged V10 shard paths, re-expressed relative to the catalog root.

The job writes per-shard receipts with exact row/token accounting and an apply
receipt with a complete per-source shard map, and never modifies the V10
caches.
"""

import argparse
import gzip
import hashlib
import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from levanter.store.cache import write_levanter_cache
from levanter.store.jagged_array import JaggedArrayStore
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer

from experiments.sft_new_mix.exp3_runtime import configure_storage, read_pinned
from experiments.sft_new_mix.loss_modes import LossMode, prediction_weights

logger = logging.getLogger(__name__)

BATCH_ROWS = 128
STORE_TOKENIZER_URI = (
    "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.15-candidate-v1/scripted-v3/tokenizer.json"
)
STORE_TOKENIZER_SHA256 = "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45"
MESSAGE_START_TOKEN = "<|start_header_id|>"
ASSISTANT_PREFIX_TEXT = "<|start_header_id|>assistant<|end_header_id|>\n"
MAX_ROW_TOKENS = 262_144
CATALOG_ROOT = "s3://marin-us-east-08a/marin/users/benfeuer"


def shard_job(source: str, entry: dict, shard_key: str, output_root: str, delta_rows: dict, catalog_root: str) -> dict:
    return {
        "source": source,
        "base_cache_dir": entry["cache_dir"].rstrip("/"),
        "base_shard_key": shard_key,
        "base_rows": entry["ledger"]["shard_rows"][shard_key],
        "base_tokens": entry["ledger"]["field_counts_by_shard"][shard_key]["input_ids"],
        "loss_mode": entry["ledger"]["metadata"]["preprocessor_metadata"]["loss_mode"],
        "deltas": delta_rows,
        "output_root": output_root.rstrip("/"),
        "catalog_root": catalog_root.rstrip("/"),
    }


def store_dir(output_root: str, source: str) -> str:
    # The V10 store-name convention: the source's '/'-separated name with '__' joins.
    return f"{output_root.rstrip('/')}/{source.replace('/', '__')}/train"


def new_shard_uri(job: dict) -> str:
    identity = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    return f"{store_dir(job['output_root'], job['source'])}/shards/{identity}/cache"


def relative_shard_key(output_cache_uri: str, output_root: str, source: str) -> str:
    prefix = store_dir(output_root, source) + "/"
    if not output_cache_uri.startswith(prefix):
        raise ValueError(f"shard {output_cache_uri} does not live under its store dir {prefix}")
    return output_cache_uri[len(prefix) :]


def root_relative_key(cache_uri: str, catalog_root: str) -> str:
    prefix = catalog_root.rstrip("/") + "/"
    if not cache_uri.startswith(prefix):
        raise ValueError(f"shard {cache_uri} does not live under the catalog root {catalog_root}")
    return cache_uri[len(prefix) :]


# Worker-process state, set by _init_worker.
_TOKENIZER: Tokenizer | None = None
_MESSAGE_START_ID: int | None = None
_ASSISTANT_PREFIX: tuple[int, ...] | None = None


def _init_worker(tokenizer_json: str) -> None:
    global _TOKENIZER, _MESSAGE_START_ID, _ASSISTANT_PREFIX
    _TOKENIZER = Tokenizer.from_str(tokenizer_json)
    _MESSAGE_START_ID = _TOKENIZER.token_to_id(MESSAGE_START_TOKEN)
    _ASSISTANT_PREFIX = tuple(_TOKENIZER.encode(ASSISTANT_PREFIX_TEXT, add_special_tokens=False).ids)
    if _MESSAGE_START_ID is None or not _ASSISTANT_PREFIX:
        raise ValueError("The store tokenizer does not render the assistant header")


def assistant_weights(tokens: np.ndarray) -> np.ndarray:
    """The source's cached supervision: predictions of the assistant bodies."""
    assert _MESSAGE_START_ID is not None and _ASSISTANT_PREFIX is not None
    return prediction_weights(
        tokens,
        loss_mode=LossMode.CHAT_ASSISTANT_ONLY,
        message_start_id=_MESSAGE_START_ID,
        assistant_prefix=_ASSISTANT_PREFIX,
        trained_assistant_turns=None,
    )


def transform_shard(job: dict) -> dict:
    """Apply the deltas for one V10 shard into the new versioned cache."""
    if job["loss_mode"] != "chat_assistant_only":
        raise ValueError(f"{job['source']}: replacement masking is defined for chat_assistant_only only")
    source = job["source"]
    part = f"{job['base_cache_dir'].rstrip('/')}/{job['base_shard_key']}"
    ids_store = JaggedArrayStore.open(f"{part}/input_ids", mode="r", dtype=np.int32)
    weights_store = JaggedArrayStore.open(f"{part}/loss_weights", mode="r", dtype=np.float32)
    if len(ids_store) != job["base_rows"] or len(weights_store) != job["base_rows"]:
        raise ValueError(f"{source} {job['base_shard_key']}: base shard row count changed")

    deltas = job["deltas"]
    lengths: list[int] = []
    supervised = 0
    replaced_tokens = 0
    replaced_original_tokens = 0
    expected_digest = hashlib.sha256()

    def records():
        nonlocal supervised, replaced_tokens, replaced_original_tokens
        for start in range(0, job["base_rows"], BATCH_ROWS):
            indices = range(start, min(start + BATCH_ROWS, job["base_rows"]))
            id_rows = ids_store.get_batch_sync(indices)
            weight_rows = weights_store.get_batch_sync(indices)
            for index, ids, weights in zip(indices, id_rows, weight_rows, strict=True):
                if index in deltas:
                    delta = deltas[index]
                    fs, key = filesystem_for(delta["payload_uri"])
                    payload = np.frombuffer(fs.cat(key), dtype="<i4")
                    if len(payload) != delta["tokens"]:
                        raise ValueError(f"{source} row {index}: payload carries {len(payload)} tokens")
                    if len(payload) > MAX_ROW_TOKENS:
                        raise ValueError(f"{source} row {index}: payload exceeds the pack cap")
                    weights = assistant_weights(payload)
                    if len(weights) != len(payload) or not weights.any():
                        raise ValueError(f"{source} row {index}: the recomputed mask is empty")
                    if int(weights.sum()) != delta["supervised_tokens"]:
                        raise ValueError(
                            f"{source} row {index}: recomputed mask supervises {int(weights.sum())} "
                            f"positions, the extract recorded {delta['supervised_tokens']}"
                        )
                    replaced_tokens += len(payload)
                    replaced_original_tokens += len(ids)
                    ids = np.asarray(payload, dtype=np.int32)
                lengths.append(len(ids))
                supervised += int(weights.sum())
                expected_digest.update(ids.astype("<i4", copy=False).tobytes())
                expected_digest.update(weights.astype("<f4", copy=False).tobytes())
                yield {"input_ids": ids, "loss_weights": weights}

    output_uri = new_shard_uri(job)
    result = write_levanter_cache(
        records(),
        output_uri,
        metadata={
            "source": source,
            "loss_mode": job["loss_mode"],
            "loss_weights_indexing": "prediction",
            "base_shard_key": job["base_shard_key"],
            "v10_1_job_sha256": hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
        },
        batch_size=BATCH_ROWS,
        max_pending_batches=2,
    )
    if result["count"] != job["base_rows"]:
        raise ValueError(f"{source} {job['base_shard_key']}: output row count differs from the base")

    output_tokens = sum(lengths)
    shard_root = output_uri.rsplit("/cache", 1)[0]
    receipt = {
        "job": job,
        "complete": True,
        "source": source,
        "output_cache_uri": output_uri,
        "input_rows": job["base_rows"],
        "input_tokens": job["base_tokens"],
        "output_rows": job["base_rows"],
        "output_tokens": output_tokens,
        "supervised_prediction_positions": supervised,
        "replaced_rows": len(deltas),
        "replaced_tokens": replaced_tokens,
        "replaced_original_tokens": replaced_original_tokens,
        "output_tokens_and_weights_sha256": expected_digest.hexdigest(),
    }
    StoragePath(f"{shard_root}/receipt.json").write_text(json.dumps(receipt, sort_keys=True))
    return {
        "shard_key": job["base_shard_key"],
        "output_cache_uri": output_uri,
        "new_shard_key": relative_shard_key(output_uri, job["output_root"], source),
        "root_relative_key": root_relative_key(output_uri, job["catalog_root"]),
        "receipt_uri": f"{shard_root}/receipt.json",
        "output_rows": job["base_rows"],
        "output_tokens": output_tokens,
        "replaced_rows": len(deltas),
        "replaced_tokens": replaced_tokens,
        "replaced_original_tokens": replaced_original_tokens,
    }


def _read_lines(path: str) -> list[str]:
    if path.endswith(".gz"):
        if "://" in path:
            fs, key = filesystem_for(path)
            return gzip.decompress(fs.cat(key)).decode().splitlines()
        with gzip.open(path, "rt") as stream:
            return stream.read().splitlines()
    if "://" in path:
        fs, key = filesystem_for(path)
        return fs.cat(key).decode().splitlines()
    return Path(path).read_text().splitlines()


def load_delta_bindings(paths: list[str], catalog: dict[str, dict]) -> dict[str, dict]:
    """Bind the stage-E delta files to sources by shard-key membership.

    The delta files carry no source field: a record belongs to the selected
    source whose ledger lists its ``shard_key`` in ``finished_shards``. Returns
    per source ``{"shards": {shard_key: {output_row: record}}, "expected_rows":
    int, "files": [paths]}``.
    """
    bindings: dict[str, dict] = {}
    total = 0
    for path in paths:
        records = [json.loads(line) for line in _read_lines(path) if line.strip()]
        owners: set[str] = set()
        seen_ids: set[str] = set()
        file_rows: dict[str, dict[int, dict]] = defaultdict(dict)
        for record in records:
            row_id = record["id"]
            if row_id in seen_ids:
                raise ValueError(f"{path}: duplicate id {row_id}")
            seen_ids.add(row_id)
            shard_key = record["shard_key"]
            matches = [source for source, entry in catalog.items() if shard_key in entry["ledger"]["finished_shards"]]
            if not matches:
                raise ValueError(f"{path}: shard_key {shard_key} is not in any selected source")
            if len(matches) > 1:
                raise ValueError(f"{path}: shard_key {shard_key} matches several sources {matches}")
            owners.update(matches)
            row = int(record["output_row"])
            if row in file_rows[shard_key]:
                raise ValueError(f"{path}: duplicate output_row {row} for shard {shard_key}")
            file_rows[shard_key][row] = {
                "payload_uri": record["payload_uri"],
                "id": row_id,
                "tokens": int(record["tokens"]),
                "supervised_tokens": int(record["supervised_tokens"]),
            }
        if len(owners) != 1:
            raise ValueError(f"{path}: records bind to {len(owners)} sources, expected exactly one")
        source = owners.pop()
        bound = bindings.setdefault(source, {"shards": defaultdict(dict), "expected_rows": 0, "files": []})
        for shard_key, rows in file_rows.items():
            clash = rows.keys() & bound["shards"][shard_key].keys()
            if clash:
                raise ValueError(f"{path}: output_rows {sorted(clash)} are already bound for {shard_key}")
            bound["shards"][shard_key].update(rows)
        bound["expected_rows"] += len(records)
        bound["files"].append(path)
        total += len(records)
    if not total:
        raise ValueError("No delta rows bound from the supplied files")
    return bindings


def apply_source(
    source: str, entry: dict, bound: dict, output_root: str, tokenizer_json: str, workers: int, catalog_root: str
) -> dict:
    """Rewrite the delta-bearing shards and map every shard of the source."""
    root = catalog_root.rstrip("/")
    ledger = entry["ledger"]
    deltas = bound["shards"]
    for shard_key, rows in deltas.items():
        if shard_key not in ledger["finished_shards"]:
            raise ValueError(f"{source}: bound shard {shard_key} is not a finished shard")
        limit = ledger["shard_rows"][shard_key]
        for output_row in rows:
            if not 0 <= output_row < limit:
                raise ValueError(f"{source} {shard_key}: output_row {output_row} is outside 0..{limit}")
    applied = sum(len(rows) for rows in deltas.values())
    if applied != bound["expected_rows"]:
        raise ValueError(f"{source}: applied {applied} delta rows, expected {bound['expected_rows']}")

    jobs = [shard_job(source, entry, shard_key, output_root, deltas[shard_key], root) for shard_key in sorted(deltas)]
    logger.info("%s: rewriting %d shards (%d delta rows)", source, len(jobs), applied)
    if workers <= 1:
        _init_worker(tokenizer_json)
        results = [transform_shard(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(tokenizer_json,)) as pool:
            results = list(pool.map(transform_shard, jobs))
    if sum(result["replaced_rows"] for result in results) != applied:
        raise ValueError(f"{source}: shard receipts account for fewer delta rows than bound")

    unchanged = []
    shard_map = {}
    for shard_key in sorted(ledger["finished_shards"]):
        if shard_key in deltas:
            result = next(result for result in results if result["shard_key"] == shard_key)
            shard_map[shard_key] = {
                "cache_uri": result["output_cache_uri"],
                "root_relative_key": result["root_relative_key"],
                "rows": result["output_rows"],
                "tokens": result["output_tokens"],
                "replaced_rows": result["replaced_rows"],
            }
        else:
            cache_uri = f"{entry['cache_dir'].rstrip('/')}/{shard_key}"
            row = {
                "shard_key": shard_key,
                "cache_uri": cache_uri,
                "root_relative_key": root_relative_key(cache_uri, root),
                "rows": ledger["shard_rows"][shard_key],
                "tokens": ledger["field_counts_by_shard"][shard_key]["input_ids"],
            }
            unchanged.append(row)
            shard_map[shard_key] = {key: row[key] for key in ("cache_uri", "root_relative_key", "rows", "tokens")}

    output_rows = sum(row["rows"] for row in shard_map.values())
    output_tokens = sum(row["tokens"] for row in shard_map.values())
    base_rows = ledger["total_num_rows"]
    base_tokens = ledger["field_counts"]["input_ids"]
    store_receipt = {
        "source": source,
        "loss_mode": entry["ledger"]["metadata"]["preprocessor_metadata"]["loss_mode"],
        "base": {
            "cache_dir": entry["cache_dir"],
            "rows": base_rows,
            "tokens": base_tokens,
            "shards": len(ledger["finished_shards"]),
        },
        "output": {
            "rows": output_rows,
            "tokens": output_tokens,
            "rewritten_shards": len(results),
            "unchanged_shards": len(unchanged),
            "row_delta": output_rows - base_rows,
            "token_delta": output_tokens - base_tokens,
        },
        "replaced_rows": applied,
        "replaced_tokens": sum(result["replaced_tokens"] for result in results),
        "replaced_original_tokens": sum(result["replaced_original_tokens"] for result in results),
        "delta_files": bound["files"],
        "shards": results,
        "unchanged_shards": unchanged,
        "shard_map": shard_map,
    }
    if output_rows != base_rows:
        raise ValueError(f"{source}: row population does not conserve")
    logger.info(
        "%s: rows %d -> %d, tokens %d -> %d, replaced %d rows",
        source,
        base_rows,
        output_rows,
        base_tokens,
        output_tokens,
        applied,
    )
    return store_receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-uri", required=True)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument(
        "--delta-bindings",
        action="append",
        default=[],
        help="A stage-E delta.jsonl(.gz) with resolved payload_uri per row; repeatable",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--catalog-root", default=CATALOG_ROOT, help="The root the receipt's keys are relative to")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()

    catalog_raw = read_pinned(args.catalog_uri, args.catalog_sha256)
    if not catalog_raw:
        raise ValueError(f"catalog read returned empty bytes from {args.catalog_uri}")
    catalog = json.loads(catalog_raw)["splits"]["train"]
    sources = set(args.source)
    for source in sources:
        if source not in catalog:
            raise ValueError(f"{source} is not in the V10 catalog")
    selected = {source: catalog[source] for source in sorted(sources)}
    bindings = load_delta_bindings(args.delta_bindings, selected)

    tokenizer_json = read_pinned(STORE_TOKENIZER_URI, STORE_TOKENIZER_SHA256).decode()
    receipt = {
        "status": "COMPLETE",
        "stage": "V10.1 toolconv apply (V10 assistant-only supervision preserved)",
        "pins": {
            "catalog_uri": args.catalog_uri,
            "catalog_sha256": args.catalog_sha256,
            "delta_bindings": list(args.delta_bindings),
            "output_root": args.output_root,
            "tokenizer_uri": STORE_TOKENIZER_URI,
            "tokenizer_sha256": STORE_TOKENIZER_SHA256,
            "catalog_root": args.catalog_root,
        },
        "sources": {},
    }
    for source in sorted(sources):
        receipt["sources"][source] = apply_source(
            source,
            selected[source],
            bindings[source],
            args.output_root,
            tokenizer_json,
            args.workers,
            args.catalog_root,
        )

    document = json.dumps(receipt, sort_keys=True, indent=1).encode()
    fs, key = filesystem_for(f"{args.destination.rstrip('/')}/apply-receipt.json")
    fs.pipe(key, document)
    print(
        json.dumps(
            {
                source: {
                    "rows": f'{data["base"]["rows"]}->{data["output"]["rows"]}',
                    "tokens": f'{data["base"]["tokens"]}->{data["output"]["tokens"]}',
                    "replaced_rows": data["replaced_rows"],
                    "rewritten_shards": data["output"]["rewritten_shards"],
                    "unchanged_shards": data["output"]["unchanged_shards"],
                }
                for source, data in receipt["sources"].items()
            },
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
