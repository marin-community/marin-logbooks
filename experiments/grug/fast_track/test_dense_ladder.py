# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.grug.fast_track.dense_ladder import ladder_rung
from experiments.grug.fast_track.train import _compute_flops


@pytest.mark.parametrize("budget", [1e18, 1.7e19, 1e21])
def test_dense_ladder_spends_budget_without_overshoot(budget):
    # Check against the trainer's FLOP currency, including head and attention.
    # One extra update must exceed the budget; no partial update is permitted.
    rung = ladder_rung(budget)
    flops_per_example, _ = _compute_flops(model_config=rung.model)
    flops_per_step = flops_per_example * rung.batch_size
    realized_flops = flops_per_step * rung.steps
    assert 0 < realized_flops <= budget < realized_flops + flops_per_step
