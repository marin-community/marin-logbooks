# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import itertools

import numpy as np
import pytest

from experiments.grug_sft.reclose_eval import close_count_tail


def brute_force_tail(p: np.ndarray) -> tuple[float, float]:
    at_least_one = at_least_two = 0.0
    for outcome in itertools.product((0, 1), repeat=len(p)):
        probability = float(np.prod([q if hit else 1.0 - q for q, hit in zip(p, outcome, strict=True)]))
        at_least_one += probability * (sum(outcome) >= 1)
        at_least_two += probability * (sum(outcome) >= 2)
    return at_least_one, at_least_two


@pytest.mark.parametrize(
    "p",
    [
        np.array([0.01, 0.2, 0.0, 0.5, 0.03, 0.9, 0.001, 0.4]),
        np.array([1.0, 0.0, 0.3]),
        np.array([1.0, 1.0, 0.0]),
        np.array([0.0, 0.0]),
    ],
)
def test_close_count_tail_matches_enumeration(p):
    assert close_count_tail(p) == pytest.approx(brute_force_tail(p), abs=1e-6)
