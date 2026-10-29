# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from experiments.post_training.identity_path_b.base import BASE_BINDING, check_base_exposure


def test_base_worker_refuses_trace_pool_wrong_dose_and_source_repeats():
    exposure = {
        **BASE_BINDING,
        "identity_share_target": 0.25,
        "identity_minimum_excludes_structural_tokens": 1500000,
        "batch_size": 8,
        "sequence_length": 8192,
        "final_text_tokens": {"identity": 1500001},
        "reasoning_text_tokens": {"identity": 0, "replay": 1234},
        "maximum_identity_source_occurrences_across_modes": 2,
    }
    check_base_exposure(exposure, "25")
    changes = [
        {"experiment": "B-traces"},
        {"identity_pool_sha256": "unreviewed"},
        {"final_text_tokens": {"identity": 1499999}},
        {"reasoning_text_tokens": {"identity": 8}},
        {"maximum_identity_source_occurrences_across_modes": 4},
    ]
    for change in changes:
        altered = copy.deepcopy(exposure)
        altered.update(change)
        with pytest.raises(ValueError, match=r"B-base|Worker exposure"):
            check_base_exposure(altered, "25")
    with pytest.raises(ValueError, match="share/minimum"):
        check_base_exposure(exposure, "10")
