# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""V10.2 candidate build, stage 1: extract the delta payloads and write the
apply bindings.

For every content-confirmed not-applied row: fetch its batch archive in-region,
verify the archive and export-member digests, and write the converted payload
as a store-tokenizer ``.i32`` — re-tokenizing ``text.txt`` with the store
tokenizer when the export's tokenizer differs from the store pin (the 10,654
quarantined rows' design path). Emits the delta bindings the apply consumes,
with each row's ``shard_key`` expressed in the V10.1 catalog's key space (the
math sources' keys remapped through the V10.1 apply receipt's shard map) and
``output_row`` the row's position in the sealed V10.1 shard.
"""

import argparse
import gzip
import hashlib
import json
import logging
import os
import tarfile
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer

from experiments.sft_new_mix.exp3_runtime import configure_storage, read_pinned
from experiments.sft_new_mix.loss_modes import LossMode, prediction_weights
from experiments.sft_new_mix.v10_1_toolconv_apply import ASSISTANT_PREFIX_TEXT, MESSAGE_START_TOKEN

logger = logging.getLogger(__name__)

STORE_TOKENIZER_URI = (
    "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.15-candidate-v1/scripted-v3/tokenizer.json"
)
STORE_TOKENIZER_SHA256 = "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45"
V10_1_APPLY_RECEIPT_URI = "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/apply-v2/apply-receipt.json"
V10_1_APPLY_RECEIPT_SHA256 = "041fffcad2d74f2f268bf59a318d66302d788b767a58acdffe608a003d578934"


def extract_one(task: tuple) -> dict:
    """Fetch one row's archive, verify, and write its payload; returns the delta record."""
    row, tokenizer, output_prefix, work_dir, seen, range_map = task
    counts: Counter = Counter()
    try:
        payload_uri = f"{output_prefix.rstrip('/')}/payloads/{row['source'].replace('/', '__')}/{row['id']}.i32"
        try:
            fs_payload, key_payload = filesystem_for(payload_uri)
            if fs_payload.info(key_payload):  # idempotent resume: the payload already landed
                ids = np.frombuffer(fs_payload.cat(key_payload), dtype="<i4")
                message_start = tokenizer.token_to_id(MESSAGE_START_TOKEN)
                assistant_prefix = tuple(tokenizer.encode(ASSISTANT_PREFIX_TEXT, add_special_tokens=False).ids)
                weights = prediction_weights(
                    ids,
                    loss_mode=LossMode.CHAT_ASSISTANT_ONLY,
                    message_start_id=message_start,
                    assistant_prefix=assistant_prefix,
                    trained_assistant_turns=None,
                )
                counts["already_present"] += 1
                return {
                    "delta": {
                        "id": row["id"],
                        "source": row["source"],
                        "shard_key": row["v10_1_shard_key"],
                        "output_row": row["store_row"],
                        "payload_uri": payload_uri,
                        "tokens": len(ids),
                        "supervised_tokens": int(weights.sum()),
                        "retokenized": row["retokenize"],
                    },
                    "counts": counts,
                }
        except FileNotFoundError:
            pass
        archive_uri = row["archive_uri"]
        local = seen.get(archive_uri)
        if local is None:
            fs, key = filesystem_for(archive_uri)
            final = work_dir / (hashlib.sha256(archive_uri.encode()).hexdigest() + ".tar.gz")
            staging = work_dir / (final.name + f".{threading.get_ident()}.tmp")
            staging.write_bytes(fs.cat(key))
            if not final.exists():
                # Atomic publish: concurrent readers of `final` never see a torn file.
                os.replace(staging, final)
            local = final
            seen[archive_uri] = local
            counts["archives_fetched"] += 1
        prefix = row["record_prefix"]
        files = row["files"]

        def extract_from(local: Path) -> tuple:
            with tarfile.open(local, "r:gz") as tar:
                try:
                    ids_member = tar.extractfile(f"{prefix}export/input_ids.i32") if prefix else None
                    text_member = tar.extractfile(f"{prefix}export/text.txt") if prefix else None
                except KeyError:
                    ids_member = text_member = None
                if ids_member is None or text_member is None:
                    # The re-batched archives don't always honor the recorded prefix;
                    # locate the row's records by its id inside the tar.
                    wanted_suffix = f"/records/{row['id']}/export/"
                    match = next(
                        (name for name in tar.getnames() if name.endswith(wanted_suffix + "input_ids.i32")),
                        None,
                    )
                    if match is None:
                        raise KeyError(f"no members for {row['id']} in the archive")
                    base = match[: -len("input_ids.i32")]
                    ids_member = tar.extractfile(base + "input_ids.i32")
                    text_member = tar.extractfile(base + "text.txt")
                return ids_member.read(), text_member.read()

        try:
            ids_bytes, text_bytes = extract_from(local)
        except KeyError:
            # The RNO-era re-ingest bound some rows to batch archives that do not
            # contain them. Try the binding's candidates (same row range), then
            # the range's full archive set from the registry map.
            search_uris = list(row.get("candidate_uris") or [])
            row_range = row.get("row_range")
            if not search_uris and row_range and range_map:
                search_uris = range_map.get(row_range, [])
            if not search_uris:
                raise
            ids_bytes = text_bytes = None
            digest_matched = False

            def try_archive(candidate_uri: str) -> bool:
                nonlocal ids_bytes, text_bytes, digest_matched
                candidate = seen.get(candidate_uri)
                if candidate is None:
                    cfs, ckey = filesystem_for(candidate_uri)
                    final = work_dir / (hashlib.sha256(candidate_uri.encode()).hexdigest() + ".tar.gz")
                    staging = work_dir / (final.name + f".{threading.get_ident()}.tmp")
                    staging.write_bytes(cfs.cat(ckey))
                    if not final.exists():
                        os.replace(staging, final)
                    candidate = final
                    seen[candidate_uri] = candidate
                    counts["candidate_archives_fetched"] += 1
                try:
                    candidate_ids, candidate_text = extract_from(candidate)
                except KeyError:
                    return False
                counts["candidate_rescues"] += 1
                expected = row["files"].get("input_ids.i32")
                if expected and hashlib.sha256(candidate_ids).hexdigest() == expected:
                    ids_bytes, text_bytes = candidate_ids, candidate_text
                    digest_matched = True
                    return True
                if ids_bytes is None:
                    ids_bytes, text_bytes = candidate_ids, candidate_text
                return False

            for candidate_uri in search_uris:
                if try_archive(candidate_uri):
                    break
            if ids_bytes is None and row_range and range_map:
                # The binding's candidates were exhausted without the row; scan
                # the registered same-range archives.
                for candidate_uri in range_map.get(row_range, []):
                    if try_archive(candidate_uri):
                        break
            if ids_bytes is None:
                raise KeyError(f"no candidate archive contains {row['id']}") from None
            if not digest_matched:
                counts["digest_mismatch_rescues"] += 1
        for name, blob in (("input_ids.i32", ids_bytes), ("text.txt", text_bytes)):
            expected = files.get(name)
            if expected and hashlib.sha256(blob).hexdigest() != expected:
                return {"error": f"{row['id']}: export member {name} digest mismatch"}
        if row["retokenize"]:
            ids = np.asarray(tokenizer.encode(text_bytes.decode(), add_special_tokens=False).ids, dtype=np.int32)
            retokenized = True
        else:
            ids = np.frombuffer(ids_bytes, dtype="<i4")
            retokenized = False
        message_start = tokenizer.token_to_id(MESSAGE_START_TOKEN)
        assistant_prefix = tuple(tokenizer.encode(ASSISTANT_PREFIX_TEXT, add_special_tokens=False).ids)
        weights = prediction_weights(
            ids,
            loss_mode=LossMode.CHAT_ASSISTANT_ONLY,
            message_start_id=message_start,
            assistant_prefix=assistant_prefix,
            trained_assistant_turns=None,
        )
        if not weights.any():
            return {"error": f"{row['id']}: the recomputed assistant mask is empty"}
        fs_payload, key_payload = filesystem_for(payload_uri)
        fs_payload.pipe(key_payload, ids.astype("<i4", copy=False).tobytes())
        return {
            "delta": {
                "id": row["id"],
                "source": row["source"],
                "shard_key": row["v10_1_shard_key"],
                "output_row": row["store_row"],
                "payload_uri": payload_uri,
                "tokens": len(ids),
                "supervised_tokens": int(weights.sum()),
                "retokenized": retokenized,
            },
            "counts": counts,
        }
    except Exception as error:  # per-row isolation; every failure lands in the receipt
        return {"error": f"{row['id']}: {type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", required=True, help="The exported delta bindings (.jsonl.gz)")
    parser.add_argument("--tokenizer-uri", default=STORE_TOKENIZER_URI)
    parser.add_argument("--tokenizer-sha256", default=STORE_TOKENIZER_SHA256)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--range-candidates",
        default=None,
        help="JSON map {row_range: [archive URIs]} for rows whose bound archive lacks their record",
    )
    parser.add_argument("--fetch-workers", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()

    tokenizer = Tokenizer.from_str(read_pinned(args.tokenizer_uri, args.tokenizer_sha256).decode())
    receipt = json.loads(read_pinned(V10_1_APPLY_RECEIPT_URI, V10_1_APPLY_RECEIPT_SHA256).decode())
    shard_maps = {
        source: {key: value["root_relative_key"] for key, value in data["shard_map"].items()}
        for source, data in receipt["sources"].items()
    }

    range_map = {}
    if args.range_candidates:
        rfs, rkey = filesystem_for(args.range_candidates)
        range_map = json.loads(rfs.cat(rkey))
        logger.info("range candidates: %d ranges", len(range_map))

    fs, key = filesystem_for(args.bindings)
    rows = [json.loads(line) for line in gzip.decompress(fs.cat(key)).decode().splitlines() if line.strip()]
    logger.info("bindings: %d rows", len(rows))

    work_dir = Path(tempfile.mkdtemp(prefix="v10-2-payloads-"))
    seen: dict[str, Path] = {}
    tasks = []
    for row in rows:
        v10_1_key = shard_maps.get(row["source"], {}).get(row["store_shard"], row["store_shard"])
        tasks.append(({**row, "v10_1_shard_key": v10_1_key}, tokenizer, args.output_prefix, work_dir, seen, range_map))

    counts: Counter = Counter()
    deltas = []
    errors = []
    with ThreadPoolExecutor(max_workers=args.fetch_workers) as pool:
        for outcome in pool.map(extract_one, tasks):
            counts.update(outcome.get("counts", {}))
            if "error" in outcome:
                errors.append(outcome["error"])
            else:
                deltas.append(outcome["delta"])
    logger.info("payloads written: %d; errors: %d; %s", len(deltas), len(errors), dict(counts))
    if errors:
        for error in errors[:20]:
            logger.error("%s", error)
        raise ValueError(f"{len(errors)} payload extractions failed; first: {errors[0]}")

    by_source = Counter(delta["source"] for delta in deltas)
    retokenized = sum(1 for delta in deltas if delta["retokenized"])
    lines = "".join(json.dumps(delta, sort_keys=True) + "\n" for delta in deltas)
    bindings_uri = f"{args.output_prefix.rstrip('/')}/delta-bindings-v10-2.jsonl"
    StoragePath(bindings_uri).write_text(lines)
    extract_receipt = {
        "status": "COMPLETE",
        "stage": "V10.2 candidate payload extraction",
        "pins": {
            "bindings": args.bindings,
            "tokenizer_sha256": args.tokenizer_sha256,
            "v10_1_apply_receipt": {"uri": V10_1_APPLY_RECEIPT_URI, "sha256": V10_1_APPLY_RECEIPT_SHA256},
        },
        "rows": len(rows),
        "deltas": len(deltas),
        "by_source": dict(sorted(by_source.items())),
        "retokenized": retokenized,
        "counts": dict(counts),
    }
    document = json.dumps(extract_receipt, sort_keys=True, indent=1).encode()
    out_fs, out_key = filesystem_for(f"{args.output_prefix.rstrip('/')}/payload-extract-receipt.json")
    out_fs.pipe(out_key, document)
    summary = {k: extract_receipt[k] for k in ("rows", "deltas", "by_source", "retokenized")}
    print(json.dumps(summary, indent=1, sort_keys=True))
    print("bindings:", bindings_uri)


if __name__ == "__main__":
    main()
