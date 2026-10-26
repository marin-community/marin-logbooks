# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select a fixed, small workload from the actual Snowball RLVR1 artifact."""

import hashlib
import json
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath

SOURCE_URI = "s3://marin-us-east-02a/marin/users/dml/pivotrl/mixed-rlvr/rlvr1/2026.10.16.3"
SOURCE_ETAG = '"57b5abf0f42a342b7a06c0bfbf9af30f-4"'
AGENTS = (
    "instruction_following_simple_agent",
    "single_step_tool_use_with_argument_comparison_agent",
    "calendar_simple_agent",
)


def prepare_performance_data(source_path: str, output_path: str, smoke: bool) -> None:
    targets = dict(zip(AGENTS, (3, 1, 0) if smoke else (4, 2, 2), strict=True))
    counts, selected, source_rows = Counter(), [], []
    filesystem, source_file = filesystem_for(str(StoragePath(source_path) / "train.parquet"))
    observed_etag = filesystem.info(source_file)["ETag"]
    assert observed_etag == SOURCE_ETAG, observed_etag
    with filesystem.open(source_file, "rb") as source:
        parquet = pq.ParquetFile(source)
        source_index = 0
        for batch in parquet.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                index, source_index = source_index, source_index + 1
                agent = row["extra_info"]["nemotron_ultra"]["agent"]
                if counts[agent] >= targets.get(agent, 0):
                    continue
                if smoke and len(json.dumps(row["prompt"])) > 16000:
                    continue
                selected.append(row)
                source_rows.append(index)
                counts[agent] += 1
            if all(counts[agent] == count for agent, count in targets.items()):
                break
    assert len(selected) == sum(targets.values()), counts
    output = StoragePath(output_path)
    table = pa.Table.from_pylist(selected)
    for filename in ("train.parquet", "validation.parquet"):
        with (output / filename).open("wb") as destination:
            pq.write_table(table, destination)
    manifest = {
        "source_uri": SOURCE_URI,
        "source_etag": SOURCE_ETAG,
        "source_rows": source_rows,
        "agents": dict(counts),
        "rows": len(selected),
        "smoke": smoke,
        "row_sha256": [
            hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            for row in selected
        ],
        "limits": [
            "A fixed subset of the original RLVR1 workload; not its full task mixture.",
            "Only locally graded tasks are selected; external math-judge credentials are unavailable.",
            "The validation file repeats the workload for startup wiring only; no held-out evaluation is run.",
            "Smoke selects shorter prompts and omits calendar tasks.",
        ],
    }
    with (output / "selection.json").open("w") as destination:
        json.dump(manifest, destination, indent=2)
