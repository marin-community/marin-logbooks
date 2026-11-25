# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F10: the production key build (``marin.datakit.decon.build_keys``) at another width, as single-task phases
(``prepare``, ``count --part K --parts N``, ``choose``), for the planted check at width 16 (review points 1, 5).

    python f10_keys_width.py prepare|count|choose --items <items> --output <dir> --width 16 [--part K --parts N]
"""
import argparse
import json
import logging

from marin.datakit import decon
from rigging.filesystem.storage_path import prefix_join
from rigging.log_setup import configure_logging

from experiments.datakit.decontam.config import REFERENCE_SEED, REFERENCE_SHARDS_PER_SOURCE

logger = logging.getLogger(__name__)
ROOT = "s3://marin-us-east-02a/marin/normalized"


class _Shard:
    def __init__(self, index: int) -> None:
        self.shard_idx = index


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "count", "choose"])
    parser.add_argument("--items", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    args = parser.parse_args()
    windows_path = prefix_join(args.output, "windows.npz")
    counts_dir = prefix_join(args.output, "counts")
    if args.phase == "prepare":
        logger.info("%s", decon._prepare_windows(args.items, windows_path, args.width))
    elif args.phase == "count":
        names = json.load(open("experiments/datakit/decontam/encoder/sources.json"))
        dirs = [prefix_join(ROOT, f"{name}/outputs/main") for name in names]
        shards = decon.reference_shards(dirs, REFERENCE_SHARDS_PER_SOURCE, REFERENCE_SEED)[args.part :: args.parts]
        for result in decon._make_count(windows_path, counts_dir, args.width)(iter(shards), _Shard(args.part)):
            logger.info("%s", result)
    else:
        logger.info("%s", decon._choose_keys(windows_path, counts_dir, args.items, args.output, args.width))


if __name__ == "__main__":
    main()
