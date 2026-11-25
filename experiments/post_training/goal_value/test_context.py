# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training.goal_value.context import prune_middle


def test_middle_pruning_retains_each_decision_and_matches_online_prefixes():
    positions = list(range(7, 104, 8))
    ids = list(range(positions[-1] + 1))
    cap = 31
    marker = [-1]
    windows = list(prune_middle(ids, positions, 2, marker, cap))
    assert [p for window in windows for p in window.source_positions] == positions
    assert windows[0].ids == ids[:24]

    for window in windows[1:]:
        assert window.ids[:7] == [*ids[:6], *marker]
        assert len(window.ids) <= cap
        for position, source in zip(window.positions, window.source_positions, strict=True):
            assert window.ids[position] == ids[source]
            # Rebuilding the window before this action gives the same input.
            past = [pos for pos in positions if pos <= source]
            online = list(prune_middle(ids[: source + 1], past, 2, marker, cap))[-1]
            assert online.ids == window.ids[: position + 1]
            tail_start = online.ids[7]
            assert tail_start in [pos - 1 for pos in past[:-1]]


def test_oversized_observation_keeps_its_recent_suffix():
    ids = list(range(97))
    windows = list(prune_middle(ids, [5, 90, 96], 2, [-1, -2], 32))
    after_observation = windows[1]
    assert after_observation.ids[:6] == [0, 1, 2, 3, -1, -2]
    position = after_observation.positions[0]
    assert after_observation.ids[position - 7 : position + 1] == ids[83:91]
    assert [p for window in windows for p in window.source_positions] == [5, 90, 96]
