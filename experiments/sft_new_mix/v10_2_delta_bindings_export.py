# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the V10.2 delta bindings for the content-confirmed not-applied rows.

Reads the content-attribution result (the refined not-applied list), resolves
each row's full binding from the durable index (archive URI via the archives
table, record prefix, export member hashes, tokenizer pin), and writes one
JSONL row per delta row with everything the payload-extraction job needs.
"""

import argparse
import gzip
import json
import re
import sqlite3
import zlib

from rigging.filesystem.buckets import filesystem_for

SOURCES = (
    "nemotron_sft_v3/math_v3/train",
    "nemotron_sft_v3/math_v2/high",
    "nemotron_sft_v3/math_v2/low",
    "nemotron_sft_v3/math_v4/train",
    "nemotron_sft_v3/opencode_v1/agent_skills",
    "nemotron_sft_v3/opencode_v1/bash_only_tool_skills",
    "nemotron_sft_v3/opencode_v1/bash_only_tool",
    "nemotron_sft_v3/opencode_v1/question_tool",
    "nemotron_sft_v3/swe_v1/r2e_gym",
    "nemotron_sft_v3/swe_v2/openhands_swe",
    "nemotron_sft_v3/science_v2/rqa",
    "nemotron_sft_v3/science_v2/so",
    "nemotron_sft_v3/science_v2/syn_mcq",
    "science-tool-use-conversations",
)
STORE_TOKENIZER_SHA = "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--not-applied", required=True, help="The refined attribution's not-applied.jsonl.gz")
    parser.add_argument("--db", default="/Users/benfeuer/.local/share/marin/tool-conversion-index/index.sqlite")
    parser.add_argument("--output", required=True, help="Output .jsonl.gz path")
    args = parser.parse_args()

    fs, key = None, None
    if "://" in args.not_applied:
        fs, key = filesystem_for(args.not_applied)
        rows = [json.loads(line) for line in gzip.decompress(fs.cat(key)).decode().splitlines() if line.strip()]
    else:
        with gzip.open(args.not_applied, "rt") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    wanted = {row["id"]: row for row in rows}
    print(f"requested rows: {len(wanted)}")

    connection = sqlite3.connect(args.db)
    wanted_ids = sorted(wanted)
    bundles_needed = set()
    records = {}
    # The id index (records_id) only serves these queries when the id IN list
    # drives them; the status/modified predicates made the planner full-scan.
    for start in range(0, len(wanted_ids), 900):
        chunk = wanted_ids[start : start + 900]
        placeholders = ",".join("?" * len(chunk))
        for _row_id, source, data in connection.execute(
            f"SELECT id, source, data FROM records INDEXED BY records_id "
            f"WHERE id IN ({placeholders}) AND status='verified' AND modified=1",
            chunk,
        ):
            record = json.loads(data)
            if "source_row" in record:
                records[record["id"]] = (source, record)
            else:
                bundles_needed.add(record["_bundle_sha256"])
                records[record["id"]] = (source, None)
    for start in range(0, len(bundles_needed), 400):
        chunk = sorted(bundles_needed)[start : start + 400]
        placeholders_b = ",".join("?" * len(chunk))
        for blob in connection.execute(
            f"SELECT data FROM record_bundles WHERE archive_sha256 IN ({placeholders_b})", chunk
        ):
            bundle = json.loads(zlib.decompress(blob[0]))
            for row_id, record in bundle.items():
                if row_id in records and records[row_id][1] is None:
                    records[row_id] = (records[row_id][0], record)

    archive_uris = {}
    for start in range(0, len(wanted_ids), 900):
        chunk = wanted_ids[start : start + 900]
        placeholders = ",".join("?" * len(chunk))
        for row_id, archive_sha256 in connection.execute(
            f"SELECT id, archive_sha256 FROM records INDEXED BY records_id WHERE id IN ({placeholders})",
            chunk,
        ):
            archive_row = connection.execute("SELECT uri FROM archives WHERE sha256=?", (archive_sha256,)).fetchone()
            if archive_row:
                archive_uris[row_id] = archive_row[0]

    written = 0
    missing = 0
    with gzip.open(args.output, "wt") as stream:
        for row_id, (source, record) in records.items():
            if record is None:
                missing += 1
                continue
            manifest = record.get("export_manifest") or {}
            archive_uri = archive_uris.get(row_id)
            if not archive_uri or not manifest.get("files"):
                missing += 1
                continue
            # The RNO-era re-ingest bound some rows to batch archives that do not
            # contain them. Candidates: the registry's archives carrying the same
            # row range as the record's own prefix shard (the record lives in one
            # of them; the payload digest verifies which).
            candidate_uris = []
            row_range = None
            prefix = record.get("record_prefix", "")
            if prefix and "/run/records/" in prefix:
                shard_name = prefix.split("/run/records/", 1)[0].rsplit("/", 1)[-1]
                if shard_name:
                    row_range = re.search(r"rows-\d+-\d+", shard_name).group(0)
                    if row_range and not archive_uri.endswith(shard_name + ".tar.gz"):
                        exact = connection.execute(
                            "SELECT uri FROM archives WHERE uri LIKE ? AND uri != ? LIMIT 4",
                            (f"%{shard_name}.tar.gz", archive_uri),
                        ).fetchall()
                        if exact:
                            candidate_uris = [row[0] for row in exact]
                        else:
                            ranged = connection.execute(
                                "SELECT uri FROM archives WHERE uri LIKE ? AND uri != ? LIMIT 40",
                                (f"%{row_range}%", archive_uri),
                            ).fetchall()
                            candidate_uris = [row[0] for row in ranged]
            document = {
                "id": row_id,
                "source": source,
                "shard": record.get("source_shard", ""),
                "row": int(record["source_row"]),
                "store_shard": wanted[row_id]["store_shard"],
                "store_row": wanted[row_id]["store_row"],
                "archive_uri": archive_uri,
                "candidate_uris": candidate_uris,
                "row_range": row_range,
                "record_prefix": record["record_prefix"],
                "files": manifest["files"],
                "tokenizer_sha256": manifest.get("tokenizer_sha256"),
                "tokens": manifest.get("tokens"),
                "supervised_tokens": manifest.get("supervised_tokens"),
                "retokenize": manifest.get("tokenizer_sha256") not in (None, STORE_TOKENIZER_SHA),
            }
            stream.write(json.dumps(document, separators=(",", ":")) + "\n")
            written += 1
    retokenize = 0
    with gzip.open(args.output, "rt") as stream:
        for line in stream:
            if json.loads(line)["retokenize"]:
                retokenize += 1
    print(
        f"wrote {written} bindings to {args.output}; {missing} without a resolvable binding; "
        f"{retokenize} need re-tokenization"
    )


if __name__ == "__main__":
    main()
