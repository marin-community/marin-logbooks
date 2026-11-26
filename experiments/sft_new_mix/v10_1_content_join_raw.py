# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""V10.1 content join, raw side (J1, 02a): hash every verified tool-conversion's ORIGINAL
row by content, in the conversion fleet's own decode.

For each verified-modified conversion of the affected sources, this job reads the
conversion's ORIGINAL raw levanter-store row (``source_shard``/``source_row`` — the raw
parts live in-region on 02a), decodes it with the pinned 2026.09.18 store tokenizer
(``skip_special_tokens=False``, the conversion fleet's decode), and emits two content
keys per row:

- ``h_raw``: sha256 of the decoded text as-is — matches the V10 row decode for rows the
  typography transform did not rewrite;
- ``h_norm``: sha256 after the typography normalization (U+202F → space, U+2011 →
  hyphen) — matches the V10 row decode for rows the transform rewrote.

The companion V10-side job (08a) hashes the V10 store rows' decodes and joins on either
key. The conversion's own ``source_text_sha256`` is carried through for cross-checking.
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
from levanter.store.jagged_array import JaggedArrayStore
from rigging.filesystem.buckets import filesystem_for
from tokenizers import Tokenizer

from experiments.sft_new_mix.exp3_runtime import configure_storage, read_pinned

logger = logging.getLogger(__name__)

STORE_TOKENIZER_URI = "s3://marin-us-east-02a/marin/grug_sft/tokenizer/2026.09.18/tokenizer.json"
STORE_TOKENIZER_SHA256 = "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45"


def typography_normalize(text: str) -> str:
    return text.replace("\u202f", " ").replace("\u2011", "-")


def load_bindings(paths: list[str], sources: set[str]) -> dict[str, dict]:
    bindings: dict[str, dict] = {}
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
                continue
            row["source"] = release_source
            bindings[row["id"]] = row
    return bindings


def group_by_part(bindings: dict[str, dict]) -> dict[str, list[tuple[str, dict]]]:
    by_part: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for row_id, row in bindings.items():
        by_part[row["source_shard"]].append((row_id, row))
    return by_part


def hash_part(job: tuple[str, list[tuple[str, dict]], str]) -> dict:
    part_uri, items, tokenizer_json = job
    tokenizer = Tokenizer.from_str(tokenizer_json)
    ids_store = JaggedArrayStore.open(f"{part_uri.rstrip('/')}/input_ids", mode="r", dtype=np.int32)
    results: dict[str, dict] = {}
    for row_id, row in items:
        source_row = int(row["source_row"])
        try:
            ids = ids_store.get_batch_sync([source_row])[0]
        except Exception as error:
            results[row_id] = {"error": f"{type(error).__name__}: {error}"}
            continue
        decoded = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
        h_raw = hashlib.sha256(decoded.encode()).hexdigest()
        h_norm = hashlib.sha256(typography_normalize(decoded).encode()).hexdigest()
        results[row_id] = {
            "h_raw": h_raw,
            "h_norm": h_norm,
            "tokens": len(ids),
            "matches_source_text_sha256": row.get("source_text_sha256") in (h_raw, h_norm),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bindings", action="append", required=True, help="A verified-modified bindings JSONL(.gz); repeatable"
    )
    parser.add_argument("--source", action="append", required=True, help="The release source name (sft/source/...)")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()

    sources = set(args.source)
    bindings = load_bindings(args.bindings, sources)
    logger.info("bindings: %d conversions across %d raw parts", len(bindings), len(group_by_part(bindings)))

    by_part = group_by_part(bindings)
    jobs = [
        (part_uri, items, read_pinned(STORE_TOKENIZER_URI, STORE_TOKENIZER_SHA256).decode())
        for part_uri, items in sorted(by_part.items())
    ]
    hash_map: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for part_results in pool.map(hash_part, jobs):
            hash_map.update(part_results)

    # One output record per conversion: both content keys + the binding.
    out_lines = []
    mismatches = 0
    errors = 0
    for row_id, row in bindings.items():
        entry = hash_map.get(row_id, {})
        if "error" in entry:
            errors += 1
            row = {**row, "hash_error": entry["error"]}
        else:
            if not entry.get("matches_source_text_sha256"):
                mismatches += 1
            row = {**row, "h_raw": entry["h_raw"], "h_norm": entry["h_norm"], "raw_tokens": entry["tokens"]}
        out_lines.append(json.dumps(row, sort_keys=True))
    blob = gzip.compress("\n".join(out_lines).encode())
    fs, key = filesystem_for(f"{args.output_prefix.rstrip('/')}/content-keys.jsonl.gz")
    fs.pipe(key, blob)

    summary = {
        "conversions": len(bindings),
        "hashed": len(out_lines) - errors,
        "errors": errors,
        "source_text_sha256_mismatches": mismatches,
        "output": {
            "uri": f"{args.output_prefix.rstrip('/')}/content-keys.jsonl.gz",
            "sha256": hashlib.sha256(blob).hexdigest(),
        },
    }
    document = json.dumps(summary, sort_keys=True, indent=1).encode()
    fs, key = filesystem_for(f"{args.destination.rstrip('/')}/raw-hash-summary.json")
    fs.pipe(key, document)
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
