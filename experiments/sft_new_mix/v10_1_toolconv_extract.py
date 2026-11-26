# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""V10.1 stage E: map the verified tool-conversion deltas onto V10 store rows and extract
the converted rows' payloads in the archives' region.

The durable tool-conversion index binds verified-modified rows to (raw source shard part,
row) plus the batch archive carrying the converted document. The V10 sealed stores were
built from those same raw parts — each V10 shard's receipt pins
``job.base_shard.original_input_path`` (the raw part) and its ``decisions.jsonl`` maps
``base_row`` (the raw-part row) to ``output_row`` (the sealed-store row), with the
build-time-applied conversions recorded as ``overlays`` (``edits/<output_row>.i32``).

This job, per affected source: scans the V10 shards (receipt/decisions/applied-overlay
sets), joins the conversion bindings (rows with full payload bindings from the index
export and the complete backfill handoff), emits the DELTA — bound, retained in the V10
store, and NOT already applied — then fetches each delta row's batch archive in-region,
extracts and hash-verifies the exported members, and writes the converted row payloads
(``input_ids.i32`` re-tokenized with the store tokenizer when the export's tokenizer
differs from the store pin, plus the rendered text for evidence) as compact per-source
payload files. Rows whose export tokenizer differs from the store pin and whose payload
does not decode consistently with the exported text are QUARANTINED, never substituted.

Outputs (per source): ``delta.jsonl.gz`` (the applied delta bindings),
``payloads/<id>.i32`` (store-tokenizer token ids) and a receipt with the per-stage
counts. The companion apply job (08a) consumes these to rewrite the V10 stores.
"""

import argparse
import gzip
import hashlib
import json
import logging
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer

from experiments.sft_new_mix.exp3_runtime import configure_storage, read_pinned

logger = logging.getLogger(__name__)

STORE_TOKENIZER_URI = "s3://marin-us-east-02a/marin/grug_sft/tokenizer/2026.09.18/tokenizer.json"
STORE_TOKENIZER_SHA256 = "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45"
EXPORT_TOKENIZER_SHA256 = "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45"


def load_bindings(paths: list[str], sources: set[str], archive_region: str) -> dict[str, dict]:
    """Dedup the verified-modified bindings for the affected sources, by row id.

    The index records sources without the release's ``sft/source/`` prefix; the map
    normalizes them. ``archive_region`` filters rows to archives under that region's
    bucket so the extraction job fetches only in-region archives.
    """
    region_marker = f"s3://marin-{archive_region}/"
    bindings: dict[str, dict] = {}
    skipped_region = 0
    skipped_source = 0
    for path in paths:
        if path.endswith(".gz"):
            with gzip.open(path, "rt") as stream:
                lines = stream.read().splitlines()
        elif "://" in path:
            fs, key = filesystem_for(path)
            lines = fs.cat(key).decode().splitlines()
        else:
            lines = Path(path).read_text().splitlines()
        for line in lines:
            row = json.loads(line)
            index_source = row.get("source", "")
            release_source = next((s for s in sources if s.endswith("/" + index_source) or s == index_source), None)
            if release_source is None:
                skipped_source += 1
                continue
            uri = row.get("archive_uri", "")
            if region_marker not in uri:
                skipped_region += 1
                continue
            row["source"] = release_source
            row_id = row["id"]
            previous = bindings.get(row_id)
            if previous is None or row.get("export_manifest_sha256", "") > previous.get("export_manifest_sha256", ""):
                bindings[row_id] = row
    logger.info(
        "bindings: %d unique rows kept (%d skipped other-source, %d skipped other-region)",
        len(bindings),
        skipped_source,
        skipped_region,
    )
    return bindings


def scan_source_shards(source: str, entry: dict, fs_factory) -> dict:
    """Per V10 shard: the raw part it was built from, the raw-row → output-row map, and
    the output rows already carrying a build-time conversion overlay.

    The V10 (typography) stores hold pure token caches; the build lineage (receipt.json
    with ``job.base_shard.original_input_path`` + overlays, ``decisions.jsonl`` with
    ``base_row → output_row``) lives in the ORIGINAL build tree at
    ``s3://marin-us-east-02a/marin/users/benfeuer/<shard_key minus /cache>`` — small
    metadata files, read cross-region.
    """
    lineage_root = "s3://marin-us-east-08a/marin/users/benfeuer"
    shards = {}
    for shard_key in sorted(entry["ledger"]["finished_shards"]):
        shard_root = f"{entry['cache_dir'].rstrip('/')}/{shard_key}"
        lineage_dir = f"{lineage_root}/{shard_key.removesuffix('/cache')}"
        receipt = json.loads(StoragePath(f"{lineage_dir}/receipt.json").read_text())
        job = receipt["job"]
        base = job["base_shard"]
        raw_part = base["original_input_path"]
        applied = set()
        for overlay in job.get("overlays", []):
            applied.add(int(overlay["source_row"]))
        decisions = {}
        text = StoragePath(f"{lineage_dir}/decisions.jsonl").read_text()
        for line in text.splitlines():
            record = json.loads(line)
            if record.get("stage") == "retained":
                decisions[int(record["base_row"])] = int(record["output_row"])
        shards[shard_key] = {
            "raw_part": raw_part,
            "decisions": decisions,
            "applied_base_rows": applied,
            "output_rows": receipt["output_rows"],
        }
    return shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-uri", required=True, help="The V10 cache catalog URI")
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument(
        "--bindings", action="append", required=True, help="A verified-modified bindings JSONL(.gz); repeatable"
    )
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument(
        "--archive-region",
        required=True,
        choices=("us-east-02a", "us-east-08a", "rno2a"),
        help="Only process rows whose batch archive lives in this region's bucket",
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--destination", help="Receipt/error destination prefix (defaults to --output-prefix)")
    parser.add_argument("--work", default=None)
    parser.add_argument(
        "--fetch-workers", type=int, default=12, help="Threads for the archive fetch/extract pass (I/O bound)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()

    logger.info("stage E: reading catalog %s", args.catalog_uri)
    catalog = json.loads(read_pinned(args.catalog_uri, args.catalog_sha256).decode())["splits"]["train"]
    sources = set(args.source)
    for source in sources:
        if source not in catalog:
            raise ValueError(f"{source} is not in the V10 catalog")

    logger.info("stage E: reading bindings %s", args.bindings)
    bindings = load_bindings(args.bindings, sources, args.archive_region)
    logger.info("stage E: bindings loaded (%d rows)", len(bindings))

    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="v10-1-toolconv-"))
    work.mkdir(parents=True, exist_ok=True)
    logger.info("stage E: reading tokenizer %s", STORE_TOKENIZER_URI)
    tokenizer = Tokenizer.from_str(read_pinned(STORE_TOKENIZER_URI, STORE_TOKENIZER_SHA256).decode())

    fs, _ = filesystem_for(args.output_prefix.rstrip("/") + "/")
    seen_archives: dict[str, Path] = {}
    retokenized: set[str] = set()
    receipt = {
        "status": "COMPLETE",
        "stage": f"V10.1 stage E extract (archive region {args.archive_region})",
        "pins": {
            "catalog_uri": args.catalog_uri,
            "catalog_sha256": args.catalog_sha256,
            "bindings": [{"path": path} for path in args.bindings],
            "store_tokenizer_sha256": STORE_TOKENIZER_SHA256,
        },
        "sources": {},
    }

    for source in sorted(sources):
        entry = catalog[source]
        shards = scan_source_shards(source, entry, filesystem_for)
        by_raw_part = defaultdict(list)
        for shard_key, info in shards.items():
            by_raw_part[info["raw_part"]].append((shard_key, info))

        delta_rows = []
        counts = Counter()
        quarantined = []
        seen_archives: dict[str, Path] = {}

        for row_id, row in bindings.items():
            if row.get("source") != source:
                continue
            counts["bound_rows"] += 1
            raw_part = row.get("source_shard", "")
            raw_row = int(row["source_row"])
            matches = by_raw_part.get(raw_part, ())
            if not matches:
                counts["raw_part_not_in_v10_lineage"] += 1
                continue
            if len(matches) != 1:
                counts["raw_part_ambiguous"] += 1
                continue
            shard_key, info = matches[0]
            output_row = info["decisions"].get(raw_row)
            if output_row is None:
                counts["raw_row_not_retained"] += 1
                continue
            if raw_row in info["applied_base_rows"]:
                counts["already_applied_at_build"] += 1
                continue
            export_files = (row.get("export_manifest") or {}).get("files") or {}
            if "input_ids.i32" not in export_files or "text.txt" not in export_files:
                counts["binding_missing_export_files"] += 1
                continue
            if row.get("export_manifest", {}).get("tokenizer_sha256") not in (None, EXPORT_TOKENIZER_SHA256):
                counts["tokenizer_mismatch_quarantined"] += 1
                quarantined.append(
                    {
                        "id": row_id,
                        "reason": "export tokenizer differs from the store pin",
                        "tokenizer_sha256": row["export_manifest"].get("tokenizer_sha256"),
                    }
                )
                continue
            delta_rows.append((row_id, row, shard_key, output_row))
            counts["delta_rows"] += 1

        # Fetch each delta row's archive once; extract and validate the members.
        # The per-row work is I/O bound (one batch archive per distinct archive URI).
        payload_dir = work / f"payloads-{args.archive_region}"
        payload_dir.mkdir(parents=True, exist_ok=True)

        def fetch_one(
            item: tuple,
            _seen: dict = seen_archives,
            _dir: Path = payload_dir,
            _tok: Tokenizer = tokenizer,
            _source: str = source,
            _prefix_out: str = args.output_prefix.rstrip("/"),
        ) -> dict:
            row_id, row, _shard_key, _output_row = item
            counts_local: Counter = Counter()
            try:
                archive_uri = row["archive_uri"]
                local = _seen.get(archive_uri)
                if local is None:
                    fsa, keya = filesystem_for(archive_uri)
                    local = _dir / (hashlib.sha256(archive_uri.encode()).hexdigest() + ".tar.gz")
                    local.write_bytes(fsa.cat(keya))
                    _seen[archive_uri] = local
                    counts_local["archives_fetched"] += 1
                prefix = row["record_prefix"]
                files = row["export_manifest"]["files"]
                with tarfile.open(local, "r:gz") as tar:
                    member_ids = tar.extractfile(f"{prefix}export/input_ids.i32")
                    member_text = tar.extractfile(f"{prefix}export/text.txt")
                    if member_ids is None or member_text is None:
                        counts_local["archive_member_missing"] += 1
                        return {
                            "counts": counts_local,
                            "quarantine": {"id": row_id, "reason": "archive members missing"},
                        }
                    ids_bytes = member_ids.read()
                    text_bytes = member_text.read()
                for name, blob in (("input_ids.i32", ids_bytes), ("text.txt", text_bytes)):
                    if hashlib.sha256(blob).hexdigest() != files[name]:
                        counts_local["member_hash_mismatch"] += 1
                        return {"counts": counts_local, "quarantine": {"id": row_id, "reason": f"{name} hash mismatch"}}
                ids = np.frombuffer(ids_bytes, dtype="<i4")
                decoded = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
                exported_text = text_bytes.decode()
                if decoded != exported_text:
                    ids = np.asarray(tokenizer.encode(exported_text, add_special_tokens=False).ids, dtype=np.int32)
                    counts_local["retokenized_from_text"] += 1
                    retokenized.add(row_id)
                else:
                    counts_local["payload_ids_validated"] += 1
                payload_file = _dir / f"{row_id}.i32"
                payload_file.write_bytes(ids.astype("<i4", copy=False).tobytes())
                blob = payload_file.read_bytes()
                fs_payload, key_payload = filesystem_for(
                    f"{_prefix_out}/payloads/{_source.replace('/', '__')}/{row_id}.i32"
                )
                fs_payload.pipe(key_payload, blob)
                payload_file.unlink()
                counts_local["payloads_written"] = 1
                return {"counts": counts_local}
            except Exception as error:
                counts_local["fetch_errors"] += 1
                return {
                    "counts": counts_local,
                    "quarantine": {"id": row_id, "reason": f"{type(error).__name__}: {error}"},
                }

        with ThreadPoolExecutor(max_workers=args.fetch_workers) as pool:
            outcomes = list(pool.map(fetch_one, delta_rows))
        for outcome in outcomes:
            counts.update(outcome["counts"])
            if "quarantine" in outcome:
                quarantined.append(outcome["quarantine"])
        payloads = sum(outcome["counts"].get("payloads_written", 0) for outcome in outcomes)

        delta_document = "\n".join(
            json.dumps(
                {
                    "id": row_id,
                    "shard_key": shard_key,
                    "output_row": output_row,
                    "archive_uri": row["archive_uri"],
                    "archive_sha256": row["archive_sha256"],
                    "record_prefix": row["record_prefix"],
                    "export_manifest_sha256": row.get("export_manifest_sha256"),
                    "tokens": row["export_manifest"].get("tokens"),
                    "supervised_tokens": row["export_manifest"].get("supervised_tokens"),
                    "source_row": row["source_row"],
                    "source_shard": row["source_shard"],
                    "retokenized": row_id in retokenized,
                },
                sort_keys=True,
            )
            for row_id, row, shard_key, output_row in delta_rows
        )
        blob = gzip.compress(delta_document.encode())
        fsd, keyd = filesystem_for(f"{args.output_prefix.rstrip('/')}/{source.replace('/', '__')}/delta.jsonl.gz")
        fsd.pipe(keyd, blob)

        receipt["sources"][source] = {
            "counts": dict(counts),
            "delta_rows": len(delta_rows),
            "payloads_written": payloads,
            "quarantined": quarantined[:200],
            "quarantined_total": len(quarantined),
            "delta_jsonl_sha256": hashlib.sha256(blob).hexdigest(),
            "delta_jsonl_rows": len(delta_rows),
        }
        logger.info("%s: %s", source, dict(counts))

    document = json.dumps(receipt, sort_keys=True, indent=1).encode()
    fs, key = filesystem_for(f"{args.output_prefix.rstrip('/')}/extract-receipt-{args.archive_region}.json")
    fs.pipe(key, document)
    print(
        json.dumps(
            {
                s: {"counts": v["counts"], "payloads": v["payloads_written"], "quarantined": v["quarantined_total"]}
                for s, v in receipt["sources"].items()
            },
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback

        argv = sys.argv
        destination = argv[argv.index("--destination") + 1] if "--destination" in argv else None
        if destination:
            document = json.dumps({"traceback": traceback.format_exc(), "argv": sys.argv}, indent=1).encode()
            from rigging.filesystem.buckets import filesystem_for

            fs, key = filesystem_for(f"{destination.rstrip('/')}/error.json")
            fs.pipe(key, document)
        raise
