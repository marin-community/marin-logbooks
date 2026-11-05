# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish a quick-loop selection whose evaluation panel adds the thoughtcrime tasks.

Main publishes evaluation sources with QUICK curation, and the selection's ``evaluation_sources``
needs an admitted FULL corpus, so this test selection is assembled directly: the base selection's
``train.parquet`` is copied unchanged and the QUICK-converted thoughtcrime tasks are appended to
``validation.parquet`` and ``full_validation.parquet``.

    uv run python logbooks/25-thoughtcrime/build_panel_selection.py \\
        --thoughtcrime-tasks <QUICK output>/thoughtcrime/normalize/part-00000.parquet
"""

import json

import click
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.task_curation.inventory import (
    FULL_VALIDATION_FILENAME,
    TASK_SCHEMA,
    TASK_SPEC_COLUMN,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
)

BASE_SELECTION = "s3://marin-us-east-02a/marin/rl-data/task-selection/snowball-100k/2026.11.04"
OUTPUT_SELECTION = "s3://marin-us-east-02a/marin/rl-data/task-selection/snowball-100k-thoughtcrime/2026.11.04.1"
DATASET = "open-athena/Marin-Thoughtcrime@231751144c48b5553d5b921054b196e5e2d073f9"
NORMALIZED_TASK_COLUMN = "task_json"


@click.command(help=__doc__)
@click.option("--thoughtcrime-tasks", required=True, help="QUICK normalize parquet for the thoughtcrime source.")
def main(thoughtcrime_tasks: str) -> None:
    configure_coreweave_s3()
    fs = fsspec.filesystem("s3")
    output = OUTPUT_SELECTION.removeprefix("s3://")
    if fs.exists(f"{output}/summary.json"):
        raise click.ClickException(f"{OUTPUT_SELECTION} already exists")
    added = pq.read_table(thoughtcrime_tasks).column(NORMALIZED_TASK_COLUMN).to_pylist()
    base = BASE_SELECTION.removeprefix("s3://")
    fs.copy(f"{base}/{TRAIN_FILENAME}", f"{output}/{TRAIN_FILENAME}")
    for filename in (VALIDATION_FILENAME, FULL_VALIDATION_FILENAME):
        with fs.open(f"{base}/{filename}", "rb") as handle:
            tasks = pq.read_table(handle).column(TASK_SPEC_COLUMN).to_pylist()
        with fs.open(f"{output}/{filename}", "wb") as handle:
            pq.write_table(pa.table({TASK_SPEC_COLUMN: tasks + added}, schema=TASK_SCHEMA), handle)
        click.echo(f"{filename}: {len(tasks)} + {len(added)} tasks")
    with fs.open(f"{base}/summary.json") as handle:
        summary = json.load(handle)
    summary["derived"] = {"base_selection": BASE_SELECTION, "added_validation": {DATASET: len(added)}}
    with fs.open(f"{output}/summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    click.echo(OUTPUT_SELECTION)


if __name__ == "__main__":
    main()
