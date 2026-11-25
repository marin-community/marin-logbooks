# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F6: the excerpts of the v11 removals (cutoff 0.6) whose item holds an [asy] drawing, to measure how many
are other math problems that share only the drawing boilerplate (a stage-2 false positive seen in the misses).

    python logbooks/81-decon-stage1-keys/scripts/f6_asy_pairs.py --items <g1v7>/items --output <parquet>
"""
import argparse
import io
import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
from marin.datakit.decon_tables import parquet_files, read_frame, write_frame
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)
DURABLE = "s3://marin-us-east-02a/user/rav/projects/decon-encoder-poc/decon-poc/catalog-v11-full"
CATALOG = "s3://marin-us-east-08a/tmp/ttl=7d/rav/decon-poc/catalog-v11-full"


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    items = pd.concat([read_frame(p, ["id", "question"]) for p in parquet_files(args.items)])
    asy = set(items[items["question"].fillna("").str.contains("[asy]", regex=False)]["id"])
    removed = read_frame(f"{DURABLE}/removed-final-06.parquet")
    removed = removed[removed["item"].isin(asy)]
    logger.info("%d removal pairs with an [asy] item in %d shards", len(removed), removed["path"].nunique())

    def excerpts(group: tuple[str, pd.DataFrame]) -> pd.DataFrame:
        path, rows = group
        name = path.split("/normalized/", 1)[1].replace("/", "__")
        pairs = pq.read_table(
            io.BytesIO(StoragePath(f"{CATALOG}/pairs/{name}.parquet").read_bytes()), columns=["record", "item", "right"]
        ).to_pandas()
        return rows.merge(pairs, on=["record", "item"], how="left")

    with ThreadPoolExecutor(32) as pool:
        frames = list(pool.map(excerpts, removed.groupby("path")))
    out = pd.concat(frames, ignore_index=True)
    logger.info("%d rows, %d with an excerpt", len(out), int(out["right"].notna().sum()))
    write_frame(out, args.output)


if __name__ == "__main__":
    main()
