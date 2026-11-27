# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the durable tool-conversion index's verified-modified rows for the
V10.1 lineage reconciliation.

Reads the local SQLite index, keeps the verified rows whose tool calls were
actually modified, restricted to the sources V10 sealed, and writes one compact
JSONL row per record: source, raw source shard, source row, id, and whether the
record carries an export manifest. Rows that arrived as bundle references are
resolved against the ``record_bundles`` table (zlib JSON, ``{id: record}``).
This is index metadata (not store data); the lineage job consumes it in-region.
"""

import argparse
import gzip
import json
import sqlite3
import zlib

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="/Users/benfeuer/.local/share/marin/tool-conversion-index/index.sqlite")
    parser.add_argument("--output", required=True, help="Output .jsonl.gz path")
    args = parser.parse_args()

    connection = sqlite3.connect(args.db)
    placeholders = ",".join("?" * len(SOURCES))
    rows = connection.execute(
        f"SELECT source, data FROM records WHERE status='verified' AND modified=1 AND source IN ({placeholders})",
        SOURCES,
    ).fetchall()

    bundle_ids = {json.loads(data)["_bundle_sha256"] for source, data in rows if "source_row" not in json.loads(data)}
    bundles = {}
    for start in range(0, len(bundle_ids), 400):
        chunk = sorted(bundle_ids)[start : start + 400]
        placeholders_b = ",".join("?" * len(chunk))
        for archive_sha256, blob in connection.execute(
            f"SELECT archive_sha256, data FROM record_bundles WHERE archive_sha256 IN ({placeholders_b})", chunk
        ):
            bundles[archive_sha256] = json.loads(zlib.decompress(blob))

    written = 0
    unresolvable = 0
    with gzip.open(args.output, "wt") as stream:
        for source, data in rows:
            record = json.loads(data)
            if "source_row" not in record:
                record = bundles[record["_bundle_sha256"]][record["id"]]
            if "source_row" not in record:
                unresolvable += 1
                continue
            diff = record.get("diff") or {}
            changes = diff.get("changes") or []
            # Tool-call renames are the transform-robust conversion signature:
            # the sealed store's text still carries the original tool name when
            # the conversion was never applied, and loses it when it was.
            renamed = [c for c in changes if c.get("kind") == "renamed" and c.get("from") != c.get("to")]
            row = {
                "source": source,
                "shard": record.get("source_shard", ""),
                "row": int(record["source_row"]),
                "id": record["id"],
                "export": bool(record.get("export_manifest")),
                "renamed_from": sorted({c["from"] for c in renamed}),
                "renamed_to": sorted({c["to"] for c in renamed}),
            }
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            written += 1
    print(f"wrote {written} rows to {args.output}; {unresolvable} rows without a resolvable source row")


if __name__ == "__main__":
    main()
