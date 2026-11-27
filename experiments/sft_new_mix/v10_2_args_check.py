# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""V10.2 candidate build, stage 1b: the args-content check for the mismatch rows.

The 40,374 opencode rows whose verified modification changed only tool-call
ARGUMENTS (same tool names) cannot be classified by the rename signature. This
job fetches each row's converted document, extracts the changed calls'
arguments from the converted ``messages.json``, and checks the decoded sealed
row for those arguments: present -> the conversion reached the store
(applied_baked); absent -> not_applied (the row joins the V10.2 delta).
"""

import argparse
import gzip
import hashlib
import io
import json
import logging
import tarfile
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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


def _converted_args_fragments(record: dict, archive_blob: bytes) -> list[str]:
    """Whitespace-insensitive fragments of the converted arguments of the changed calls."""
    prefix = record["record_prefix"]
    with tarfile.open(fileobj=io.BytesIO(archive_blob), mode="r:gz") as tar:
        member = tar.extractfile(f"{prefix}export/messages.json")
        if member is None:
            return []
        messages = json.loads(member.read())
    fragments = []
    for message in messages:
        calls = (message.get("tool_calls") or []) if isinstance(message, dict) else []
        for call in calls:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str) and len(arguments) >= 24:
                fragments.append("".join(arguments.split())[:120])
    return fragments


def check_row(row: dict, archive_blob: bytes, ids_store: JaggedArrayStore, tokenizer: Tokenizer) -> dict:
    ids = ids_store.get_batch_sync([int(row["store_row"])])[0]
    text = tokenizer.decode(np.asarray(ids, dtype=np.int32).tolist(), skip_special_tokens=False)
    compact_text = "".join(text.split())
    fragments = _converted_args_fragments(row, archive_blob)
    if not fragments:
        return {**row, "classification": "mismatch", "reason": "the converted messages carry no tool-call arguments"}
    hits = sum(1 for fragment in fragments if fragment in compact_text)
    if hits == len(fragments):
        classification = "applied_baked"
        reason = ""
    elif hits == 0:
        classification = "not_applied"
        reason = ""
    else:
        classification = "mismatch"
        reason = f"{hits} of {len(fragments)} converted-argument fragments appear in the decoded row"
    return {**row, "classification": classification, "reason": reason}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mismatch", required=True, help="The attribution's mismatch.jsonl.gz (ids + store positions)")
    parser.add_argument(
        "--bindings-export",
        required=True,
        help="The stage-E bindings export on S3 (id -> archive_uri/record_prefix/files), streamed in-region",
    )
    parser.add_argument("--tokenizer-uri", default=STORE_TOKENIZER_URI)
    parser.add_argument("--tokenizer-sha256", default=STORE_TOKENIZER_SHA256)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--fetch-workers", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()

    tokenizer = Tokenizer.from_str(read_pinned(args.tokenizer_uri, args.tokenizer_sha256).decode())
    fs, key = filesystem_for(args.mismatch)
    wanted = {}
    for line in gzip.decompress(fs.cat(key)).decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        wanted[row["id"]] = row
    logger.info("mismatch rows to check: %d", len(wanted))

    # Resolve the wanted rows' bindings by streaming the stage-E export once,
    # in-region.
    bfs, bkey = filesystem_for(args.bindings_export)
    resolved = 0
    with bfs.open(bkey, "rt") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            row_id = record.get("id")
            if row_id in wanted and record.get("archive_uri") and record.get("export_manifest", {}).get("files"):
                wanted[row_id]["archive_uri"] = record["archive_uri"]
                wanted[row_id]["record_prefix"] = record["record_prefix"]
                wanted[row_id]["files"] = record["export_manifest"]["files"]
                resolved += 1
    logger.info("resolved bindings: %d of %d", resolved, len(wanted))
    rows = [row for row in wanted.values() if row.get("archive_uri") and row.get("files") and row.get("store_shard")]
    logger.info("rows with resolvable archives: %d", len(rows))
    unresolved = len(wanted) - len(rows)
    if unresolved:
        logger.warning("%d rows without bindings in the export", unresolved)

    catalog = json.loads(
        read_pinned(
            "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.23-v10-overthinking/release-v10/cache_catalog.json",
            "afd1b128e2ee7ad26c7027b347e1ef588138eecf3d678a4de242ba7fbed2f9a5",
        ).decode()
    )["splits"]["train"]
    for row in rows:
        row["catalog_entry"] = catalog[row["source"]]["cache_dir"].rstrip("/")

    work_dir = Path(tempfile.mkdtemp(prefix="v10-2-args-"))
    seen: dict[str, Path] = {}
    counts: Counter = Counter()

    def task(row: dict) -> dict:
        try:
            archive_uri = row["archive_uri"]
            local = seen.get(archive_uri)
            if local is None:
                fs, key = filesystem_for(archive_uri)
                local = work_dir / (hashlib.sha256(archive_uri.encode()).hexdigest() + ".tar.gz")
                local.write_bytes(fs.cat(key))
                seen[archive_uri] = local
                counts["archives_fetched"] += 1
            store_shard = row["store_shard"]
            if not store_shard.startswith("s3://"):
                store_shard = f"{row['catalog_entry']}/{store_shard}"
            ids_store = JaggedArrayStore.open(f"{store_shard}/input_ids", mode="r", dtype=np.int32)
            return check_row(row, local.read_bytes(), ids_store, tokenizer)
        except Exception as error:  # per-row isolation; every failure lands in the receipt
            return {**row, "classification": "mismatch", "reason": f"{type(error).__name__}: {error}"}

    results = []
    with ThreadPoolExecutor(max_workers=args.fetch_workers) as pool:
        for outcome in pool.map(task, rows):
            results.append(outcome)
    for result in results:
        result.pop("catalog_entry", None)

    tally = Counter()
    for result in results:
        tally[(result["source"], result["classification"])] += 1
    not_applied = [result for result in results if result["classification"] == "not_applied"]
    report = {
        "status": "COMPLETE",
        "stage": "V10.2 candidate args-content check",
        "rows": len(rows),
        "tally": {f"{source}|{classification}": count for (source, classification), count in sorted(tally.items())},
        "counts": dict(counts),
    }
    document = json.dumps(report, sort_keys=True, indent=1).encode()
    out_fs, out_key = filesystem_for(f"{args.output_prefix.rstrip('/')}/args-check-receipt.json")
    out_fs.pipe(out_key, document)
    with gzip.open(work_dir / "args-not-applied.jsonl.gz", "wt") as stream:
        for result in not_applied:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
    StoragePath(f"{args.output_prefix.rstrip('/')}/args-not-applied.jsonl.gz").write_bytes(
        (work_dir / "args-not-applied.jsonl.gz").read_bytes()
    )
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
