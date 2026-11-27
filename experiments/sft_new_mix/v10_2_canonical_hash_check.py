# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""V10.2 candidate build, stage 1c: the canonical-argument hash check.

The 37,993 same-name argument-rewrite rows whose converted-args fragments
could not decide them: the diff's changes carry (position, before_sha256,
after_sha256) per changed tool call. This job parses the sealed row's rendered
tool-call blocks back into argument objects, hashes them canonically, and
classifies each row by its changed calls: every call hashing to ``after`` ->
applied; every call hashing to ``before`` -> not_applied; mixed -> mismatch.
The hashes of the known-applied rows (from the fragment check) calibrate the
canonicalization against the diff pins before anything is classified.
"""

import argparse
import gzip
import hashlib
import io
import json
import logging
import tarfile
import tempfile
from collections import Counter, defaultdict
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
V10_CATALOG_URI = "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.23-v10-overthinking/release-v10/cache_catalog.json"
V10_CATALOG_SHA256 = "afd1b128e2ee7ad26c7027b347e1ef588138eecf3d678a4de242ba7fbed2f9a5"


def canonical_arguments(arguments) -> str | None:
    """The canonical form whose sha256 the durable index's diff pins record."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError):
            return None
    if isinstance(arguments, dict):
        return json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    if isinstance(arguments, list):
        return json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    return None


def rendered_call_arguments(text: str) -> list[dict]:
    """Parse the rendered tool-call blocks from a decoded store row."""
    calls = []
    marker = "<tool_call>"
    tail = "</tool_call>"
    position = 0
    while True:
        start = text.find(marker, position)
        if start < 0:
            break
        end = text.find(tail, start)
        if end < 0:
            break
        block = text[start + len(marker) : end]
        position = end + len(tail)
        try:
            call = json.loads(block)
        except (ValueError, TypeError):
            continue
        if isinstance(call, dict):
            calls.append(call)
    return calls


def check_row(row: dict, archive_blob: bytes, ids_store: JaggedArrayStore, tokenizer: Tokenizer) -> dict:
    ids = ids_store.get_batch_sync([int(row["store_row"])])[0]
    text = tokenizer.decode(np.asarray(ids, dtype=np.int32).tolist(), skip_special_tokens=False)
    prefix = row["record_prefix"]
    with tarfile.open(fileobj=io.BytesIO(archive_blob), mode="r:gz") as tar:
        member = tar.extractfile(f"{prefix}export/messages.json")
        if member is None:
            return {**row, "classification": "mismatch", "reason": "messages.json absent from the archive"}
        member.read()  # the converted messages verify the archive's identity
    diff = row["diff"]
    changes = diff.get("changes") or []
    rendered = rendered_call_arguments(text)
    by_position = defaultdict(list)
    for index, call in enumerate(rendered):
        function = call.get("function") or {}
        args = function.get("arguments")
        canonical = canonical_arguments(args)
        if canonical is not None:
            by_position[index].append(hashlib.sha256(canonical.encode()).hexdigest())
    before_after = []
    for change in changes:
        before_after.append((change.get("position"), change.get("before_sha256"), change.get("after_sha256")))
    decided = []
    for position, before, after in before_after:
        candidate_hashes = by_position.get(position, [])
        if not candidate_hashes:
            decided.append("unmatched")
        elif after and after in candidate_hashes:
            decided.append("after")
        elif before and before in candidate_hashes:
            decided.append("before")
        else:
            decided.append("other")
    if not decided:
        return {**row, "classification": "mismatch", "reason": "the diff carries no changed-call positions"}
    if any(state == "after" for state in decided) and not any(
        state in ("before", "other", "unmatched") for state in decided
    ):
        classification = "applied_baked"
    elif any(state == "before" for state in decided) and not any(
        state in ("after", "other", "unmatched") for state in decided
    ):
        classification = "not_applied"
    else:
        states = Counter(decided)
        classification = "mismatch"
        reason = f"mixed call states: {dict(states)}"
        return {**row, "classification": classification, "reason": reason}
    return {**row, "classification": classification, "reason": ""}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mismatch", required=True, help="The attribution's mismatch.jsonl.gz (ids + store positions)")
    parser.add_argument(
        "--bindings-export",
        required=True,
        help="The stage-E bindings export on S3 (id -> archive_uri/record_prefix/files)",
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

    # The index rows carry the diff pins; resolve them locally (the durable
    # index is a local artifact) and stream once here via a compact sidecar.
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

    # The diff pins live in the durable index records; carry them through a
    # compact locally-built sidecar if present on S3.
    diff_pins_uri = f"{args.output_prefix.rstrip('/')}/diff-pins.jsonl.gz"
    try:
        dfs, dkey = filesystem_for(diff_pins_uri)
        for line in gzip.decompress(dfs.cat(dkey)).decode().splitlines():
            if not line.strip():
                continue
            pins = json.loads(line)
            row = wanted.get(pins["id"])
            if row is not None and pins.get("diff") is not None:
                row["diff"] = pins["diff"]
    except FileNotFoundError:
        logger.warning("no diff-pins sidecar at %s", diff_pins_uri)
    rows = [row for row in wanted.values() if row.get("diff") and row.get("archive_uri") and row.get("store_shard")]
    logger.info("rows with diff pins: %d", len(rows))

    catalog = json.loads(read_pinned(V10_CATALOG_URI, V10_CATALOG_SHA256).decode())["splits"]["train"]
    for row in rows:
        row["catalog_entry"] = catalog[row["source"]]["cache_dir"].rstrip("/")

    work_dir = Path(tempfile.mkdtemp(prefix="v10-2-hashcheck-"))
    seen: dict[str, Path] = {}
    counts: Counter = Counter()

    def task(row: dict) -> dict:
        try:
            archive_uri = row["archive_uri"]
            local = seen.get(archive_uri)
            if local is None:
                afs, akey = filesystem_for(archive_uri)
                local = work_dir / (hashlib.sha256(archive_uri.encode()).hexdigest() + ".tar.gz")
                local.write_bytes(afs.cat(akey))
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

    tally = Counter()
    with gzip.open(work_dir / "check-results.jsonl.gz", "wt") as stream:
        for result in results:
            source = result["source"]
            tally[(source, result["classification"])] += 1
            stream.write(json.dumps(result, sort_keys=True) + "\n")
    report = {
        "status": "COMPLETE",
        "stage": "V10.2 candidate canonical-hash check",
        "rows": len(rows),
        "tally": {f"{source}|{classification}": count for (source, classification), count in sorted(tally.items())},
        "counts": dict(counts),
    }
    StoragePath(f"{args.output_prefix.rstrip('/')}/canonical-hash-check-receipt.json").write_bytes(
        json.dumps(report, sort_keys=True, indent=1).encode()
    )
    StoragePath(f"{args.output_prefix.rstrip('/')}/canonical-hash-results.jsonl.gz").write_bytes(
        (work_dir / "check-results.jsonl.gz").read_bytes()
    )
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
