# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""V10.1 content-level conversion attribution (the 8:58 ruling).

``retained_survivor`` is not a final bucket: for every survivor the sealed
store row's decoded text is checked against the index diff's tool-call rename
names (transform-robust: the scripted transforms never rename tools):

- ``applied_baked`` — every renamed-from tool name is gone from the sealed row:
  the conversion reached the store (baked in by the campaign's bulk pass or
  substituted by a build overlay),
- ``not_applied`` — a renamed-from name is still present and its replacement is
  absent: the row is the real V10.2 delta,
- both present, or neither present, or the diff carries no renames —
  ``mismatch`` with a reason, investigated per source.

The job streams only the survivor rows' ids from the sealed stores in-region
and writes the per-source table plus the not-applied delta list.
"""

import argparse
import gzip
import json
import logging
import tempfile
from collections import Counter, defaultdict
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

# Worker-process state, set by _init_worker.
_TOKENIZER: Tokenizer | None = None


def _init_worker(tokenizer_json: str) -> None:
    global _TOKENIZER
    _TOKENIZER = Tokenizer.from_str(tokenizer_json)


def classify_rows(task: tuple[str, list[dict]]) -> list[dict]:
    """Decode one shard's survivor rows and classify them by rename presence
    and the tool-call section's presence."""
    shard_key, rows = task
    results = []
    ids_store = JaggedArrayStore.open(f"{shard_key}/input_ids", mode="r", dtype=np.int32)
    for row in rows:
        ids = ids_store.get_batch_sync([row["store_row"]])[0]
        text = _TOKENIZER.decode(np.asarray(ids, dtype=np.int32).tolist(), skip_special_tokens=False)
        from_names = row["renamed_from"]
        to_names = row["renamed_to"]
        from_present = [name for name in from_names if name in text]
        to_present = [name for name in to_names if name in text]
        if not from_names:
            classification = "mismatch"
            reason = "diff carries no tool-call renames (same-name argument rewrites)"
        elif from_present:
            # The original tool name survives in the sealed row: never applied.
            classification = "not_applied"
            reason = ""
        elif "<tool_call>" not in text:
            # The build's scripted transforms removed the tool-call section
            # entirely: the sealed row has no conversion target.
            classification = "transformed_no_tool_block"
            reason = ""
        elif to_present:
            classification = "applied_baked"
            reason = ""
        else:
            classification = "mismatch"
            reason = "the converted tool names are absent from a row that still renders a tool-call section"
        results.append(
            {
                "source": row["source"],
                "id": row["id"],
                "part": row["part"],
                "row": row["row"],
                "store_shard": shard_key,
                "store_row": row["store_row"],
                "classification": classification,
                "reason": reason,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survivors", required=True, help="The tracer's survivors.jsonl.gz")
    parser.add_argument("--index-rows", required=True, help="The index row export with rename names (.jsonl.gz)")
    parser.add_argument(
        "--catalog-uri", required=True, help="The V10 catalog: cache_dir + shard keys resolve the stores"
    )
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mismatch-output", action="store_true", help="Also write the mismatch rows for the args check")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()

    catalog = json.loads(read_pinned(args.catalog_uri, args.catalog_sha256).decode())["splits"]["train"]
    store_uri = {
        source: {
            shard_key: f"{entry['cache_dir'].rstrip('/')}/{shard_key}"
            for shard_key in entry["ledger"]["finished_shards"]
        }
        for source, entry in catalog.items()
    }

    fs, key = filesystem_for(args.index_rows)
    names = {}
    for line in gzip.decompress(fs.cat(key)).decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        names[row["id"]] = (row.get("renamed_from") or [], row.get("renamed_to") or [])

    fs, key = filesystem_for(args.survivors)
    survivors = []
    for line in gzip.decompress(fs.cat(key)).decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        renamed_from, renamed_to = names.get(row["id"], ([], []))
        survivors.append(
            {
                "source": row["source"],
                "id": row["id"],
                "part": row["part"],
                "row": row["row"],
                "store_shard": row.get("store_shard"),
                "store_row": row.get("store_row"),
                "renamed_from": renamed_from,
                "renamed_to": renamed_to,
            }
        )
    logger.info("survivors: %d", len(survivors))

    by_shard = defaultdict(list)
    unpositioned = []
    for row in survivors:
        if row["store_shard"] is None:
            unpositioned.append(row)
            continue
        shard_uri = store_uri.get(row["source"], {}).get(row["store_shard"])
        if shard_uri is None:
            unpositioned.append(row)
            continue
        by_shard[shard_uri].append(row)
    if unpositioned:
        logger.warning("%d survivors lack store positions", len(unpositioned))

    tasks = [(shard_key, rows) for shard_key, rows in sorted(by_shard.items())]
    tokenizer_json = read_pinned(STORE_TOKENIZER_URI, STORE_TOKENIZER_SHA256).decode()
    if args.workers <= 1:
        _init_worker(tokenizer_json)
        results = [result for task in tasks for result in classify_rows(task)]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(tokenizer_json,)) as pool:
            results = [result for batch in pool.map(classify_rows, tasks) for result in batch]
    results.extend(
        {
            "source": row["source"],
            "id": row["id"],
            "part": row["part"],
            "row": row["row"],
            "store_shard": None,
            "store_row": None,
            "classification": "mismatch",
            "reason": "no sealed-store position recorded for this row",
        }
        for row in unpositioned
    )

    work = Path(tempfile.mkdtemp(prefix="v10-1-attribute-"))
    tally = defaultdict(Counter)
    mismatch_reasons = defaultdict(Counter)
    with (
        gzip.open(work / "not-applied.jsonl.gz", "wt") as not_applied,
        gzip.open(work / "mismatch.jsonl.gz", "wt") as mismatch_stream,
    ):
        for result in results:
            source = result["source"]
            tally[source][result["classification"]] += 1
            if result["classification"] == "not_applied":
                not_applied.write(json.dumps(result, sort_keys=True) + "\n")
            elif result["classification"] == "mismatch":
                mismatch_reasons[source][result["reason"] or "unspecified"] += 1
                if args.mismatch_output:
                    mismatch_stream.write(json.dumps(result, sort_keys=True) + "\n")
    if args.mismatch_output:
        StoragePath(f"{args.output_prefix.rstrip('/')}/mismatch.jsonl.gz").write_bytes(
            (work / "mismatch.jsonl.gz").read_bytes()
        )

    if args.mismatch_output:
        StoragePath(f"{args.output_prefix.rstrip('/')}/mismatch.jsonl.gz").write_bytes(
            (work / "mismatch.jsonl.gz").read_bytes()
        )
    report = {
        "sources": {
            source: {
                "survivors": sum(counter.values()),
                "tally": dict(sorted(counter.items())),
                "mismatch_reasons": dict(mismatch_reasons.get(source, {})),
            }
            for source, counter in sorted(tally.items())
        },
        "totals": {
            classification: sum(counter[classification] for counter in tally.values())
            for classification in ("applied_baked", "not_applied", "transformed_no_tool_block", "mismatch")
        },
    }
    not_applied_uri = f"{args.output_prefix.rstrip('/')}/not-applied.jsonl.gz"
    StoragePath(not_applied_uri).write_bytes((work / "not-applied.jsonl.gz").read_bytes())
    document = json.dumps(report, sort_keys=True, indent=1).encode()
    out_fs, out_key = filesystem_for(f"{args.output_prefix.rstrip('/')}/attribution.json")
    out_fs.pipe(out_key, document)
    print(json.dumps(report["totals"], sort_keys=True))
    for source, data in report["sources"].items():
        print(source, json.dumps(data, sort_keys=True))


if __name__ == "__main__":
    main()
