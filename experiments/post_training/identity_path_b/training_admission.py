# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep B-base/smoke admission unchanged and admit the separately reviewed trace variant."""

from experiments.post_training.identity_path_b.smoke import check_training_exposure as check_base_training_exposure
from experiments.post_training.identity_path_b.trace_binding import check_trace_exposure


def check_training_exposure(exposure: dict, arm: str) -> None:
    if exposure.get("experiment") == "B-traces":
        check_trace_exposure(exposure, arm)
    else:
        check_base_training_exposure(exposure, arm)
