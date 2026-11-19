# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the Open-SWE outcome-ablation membership and the requested arms.

This entry point is separate from :mod:`experiments.open_swe_outcome.arms` so the functions
Zephyr ships to workers live in an importable module, not in ``__main__``.
"""

import argparse
import logging

from marin.execution.step_runner import StepRunner
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.open_swe_outcome.arms import ARMS, arm_step, index_step, membership_step, raw_partitions

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="*", choices=ARMS, default=[], help="Arms to materialize after the membership.")
    parser.add_argument("--max-concurrent", type=int, default=4)
    args = parser.parse_args()

    configure_logging(logging.INFO)
    configure_coreweave_s3()
    partitions = raw_partitions()
    membership = membership_step(index_step(partitions))
    steps = [membership, *(arm_step(arm, membership, partitions) for arm in args.arms)]
    StepRunner().run(steps, max_concurrent=args.max_concurrent)
    for step in steps:
        logger.info("%s -> %s", step.name, step.output_path)


if __name__ == "__main__":
    main()
