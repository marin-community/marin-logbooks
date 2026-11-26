# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""V10.1 content join, V10 side + join (J2, 08a): hash every V10 store row's decode and
join against the raw-side content keys.

Streams each affected V10 store shard, decodes every row with the pinned store
tokenizer (skip_special_tokens=False), and emits ``sha256(decoded)`` per row — the key
that matches the raw-side ``h_raw`` (rows the typography transform did not rewrite) or
``h_norm`` (rows it rewrote, since the transform's U+202F → space / U+2011 → hyphen
substitutions are exactly the normalization the raw side applies). Joins against the
raw-side content-keys map (excluding already-applied rows and the stage-E delta), and
writes the content-joined conversion bindings as delta records with resolved payload
URIs — the same shape the stage-E apply consumes.
"""

import argparse
import gzip
import hashlib
import json
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from levanter.store.jagged_array import JaggedArrayStore
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer

from experiments.sft_new_mix.exp3_runtime import configure_storage, read_pinned

logger = logging.getLogger(__name__)

STORE_TOKENIZER_URI = (
    "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.15-candidate-v1/scripted-v3/tokenizer.json"
)
STORE_TOKENIZER_SHA256 = "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45"


def typography_normalize(text: str) -> str:
    return text.replace("\u202f", " ").replace("\u2011", "-")


def load_raw_keys(uri: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """The raw-side content-keys map: {h_raw: binding} union {h_norm: binding}."""
    fs, key = filesystem_for(uri)
    blob = fs.cat(key)
    h_raw_map: dict[str, dict] = {}
    h_norm_map: dict[str, dict] = {}
    for line in gzip.decompress(blob).decode().splitlines():
        row = json.loads(line)
        binding = {
            "id": row["id"],
            "source": row["source"],
            "source_row": row.get("source_row"),
            "source_shard": row.get("source_shard"),
            "archive_uri": row.get("archive_uri"),
            "archive_sha256": row.get("archive_sha256"),
            "record_prefix": row.get("record_prefix"),
            "export_manifest": row.get("export_manifest"),
            "export_manifest_sha256": row.get("export_manifest_sha256"),
        }
        if "h_raw" in row:
            h_raw_map.setdefault(row["h_raw"], binding)
        if "h_norm" in row:
            h_norm_map.setdefault(row["h_norm"], binding)
    return h_raw_map, h_norm_map


def scan_applied(source: str, entry: dict) -> dict[str, set[int]]:
    """Per shard: the output rows already carrying a build-time conversion overlay."""
    applied = {}
    for shard_key in sorted(entry["ledger"]["finished_shards"]):
        lineage_dir = f"s3://marin-us-east-08a/marin/users/benfeuer/{shard_key.removesuffix('/cache')}"
        receipt = json.loads(StoragePath(f"{lineage_dir}/receipt.json").read_text())
        decisions = {}
        for line in StoragePath(f"{lineage_dir}/decisions.jsonl").read_text().splitlines():
            record = json.loads(line)
            if record.get("stage") == "retained":
                decisions[int(record["base_row"])] = int(record["output_row"])
        applied_rows = {
            decisions[int(o["source_row"])]
            for o in receipt["job"].get("overlays", [])
            if int(o["source_row"]) in decisions
        }
        applied[shard_key] = applied_rows
    return applied


def hash_shard(job: tuple[str, str, str]) -> dict[str, dict]:
    shard_key, cache_dir, tokenizer_json = job
    tokenizer = Tokenizer.from_str(tokenizer_json)
    part = f"{cache_dir.rstrip('/')}/{shard_key}"
    ids_store = JaggedArrayStore.open(f"{part}/input_ids", mode="r", dtype=np.int32)
    rows = {}
    for start in range(0, len(ids_store), 64):
        indices = range(start, min(start + 64, len(ids_store)))
        id_rows = ids_store.get_batch_sync(indices)
        for index, ids in zip(indices, id_rows, strict=True):
            decoded = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
            rows[index] = {"h": hashlib.sha256(decoded.encode()).hexdigest()}
    return {shard_key: rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-uri", required=True)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--raw-keys-uri", required=True, help="J1's content-keys.jsonl.gz")
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument(
        "--exclude-delta", action="append", default=[], help="A stage-E delta.jsonl(.gz) whose ids are already extracted"
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()

    catalog = json.loads(read_pinned(args.catalog_uri, args.catalog_sha256).decode())["splits"]["train"]
    tokenizer = Tokenizer.from_str(read_pinned(STORE_TOKENIZER_URI, STORE_TOKENIZER_SHA256).decode())
    h_raw_map, h_norm_map = load_raw_keys(args.raw_keys_uri)
    logger.info("raw keys: %d h_raw + %d h_norm", len(h_raw_map), len(h_norm_map))

    excluded: set[str] = set()
    for path in args.exclude_delta:
        if path.endswith(".gz"):
            if "://" in path:
                fs, key = filesystem_for(path)
                lines = gzip.decompress(fs.cat(key)).decode().splitlines()
            else:
                with gzip.open(path, "rt") as stream:
                    lines = stream.read().splitlines()
        elif "://" in path:
            fs, key = filesystem_for(path)
            lines = fs.cat(key).decode().splitlines()
        else:
            lines = Path(path).read_text().splitlines()
        for line in lines:
            excluded.add(json.loads(line)["id"])
    logger.info("excluded (stage-E already extracted): %d", len(excluded))

    receipt = {
        "status": "COMPLETE",
        "stage": "V10.1 content join (V10 side)",
        "pins": {
            "catalog_uri": args.catalog_uri,
            "catalog_sha256": args.catalog_sha256,
            "raw_keys_uri": args.raw_keys_uri,
        },
        "sources": {},
    }
    joined_out = []

    for source in sorted(set(args.source)):
        entry = catalog[source]
        applied = scan_applied(source, entry)
        applied_total = sum(len(rows) for rows in applied.values())

        shards = sorted(entry["ledger"]["finished_shards"])
        jobs = [(shard_key, entry["cache_dir"], tokenizer.to_str()) for shard_key in shards]
        v10_map: dict[str, tuple[str, int]] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for shard_rows in pool.map(hash_shard, jobs):
                for shard_key, rows in shard_rows.items():
                    for index, entry_row in rows.items():
                        v10_map[entry_row["h"]] = (shard_key, index)
        logger.info("%s: %d V10 rows hashed", source, len(v10_map))

        joined = []
        counts: Counter = Counter()
        for key, binding in h_raw_map.items():
            if binding["source"] != source:
                continue
            counts["conversions"] += 1
            if binding["id"] in excluded:
                counts["already_extracted"] += 1
                continue
            hit = v10_map.get(key) or v10_map.get(h_norm_map.get(key, {}).get("h_norm", ""))
            if hit is None:
                counts["no_v10_match"] += 1
                continue
            shard_key, output_row = hit
            if output_row in applied.get(shard_key, set()):
                counts["already_applied_at_build"] += 1
                continue
            counts["content_joined"] += 1
            record = {
                "id": binding["id"],
                "source": source,
                "shard_key": shard_key,
                "output_row": output_row,
                "archive_uri": binding["archive_uri"],
                "archive_sha256": binding["archive_sha256"],
                "record_prefix": binding["record_prefix"],
                "export_manifest": binding["export_manifest"],
                "export_manifest_sha256": binding["export_manifest_sha256"],
                "payload_uri": (
                    f"{args.output_prefix.rstrip('/')}/payloads/{source.replace('/', '__')}/{binding['id']}.i32"
                ),
                "join": "content",
            }
            joined.append(record)
            joined_out.append(json.dumps(record, sort_keys=True))

        receipt["sources"][source] = {
            "counts": dict(counts),
            "applied_rows": applied_total,
            "content_joined": len(joined),
        }
        logger.info("%s: %s", source, dict(counts))

    blob = gzip.compress("\n".join(joined_out).encode())
    fs, key = filesystem_for(f"{args.output_prefix.rstrip('/')}/content-joined-delta.jsonl.gz")
    fs.pipe(key, blob)
    receipt["content_joined_total"] = str(len(joined_out))
    receipt["content_joined_sha256"] = hashlib.sha256(blob).hexdigest()

    document = json.dumps(receipt, sort_keys=True, indent=1).encode()
    fs, key = filesystem_for(f"{args.destination.rstrip('/')}/content-join-receipt.json")
    fs.pipe(key, document)
    print(json.dumps({s: v["counts"] for s, v in receipt["sources"].items()}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
