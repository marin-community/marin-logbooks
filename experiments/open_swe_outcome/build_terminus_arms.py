# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the Terminus-2-format arms of the Open-SWE outcome ablation.

This entry point is separate from :mod:`experiments.open_swe_outcome.terminus_arms` so the
functions Zephyr ships to workers live in an importable module, not in ``__main__``.
"""

import logging

from marin.execution.step_runner import StepRunner
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.open_swe_outcome.terminus_arms import terminus_arm_steps

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    steps = list(terminus_arm_steps().values())
    StepRunner().run(steps)
    for step in steps:
        logger.info("%s -> %s", step.name, step.output_path)


if __name__ == "__main__":
    main()
