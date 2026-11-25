# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan bounded causal windows while keeping the task and recent observations."""

from bisect import bisect_left
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum


class ContextPolicy(StrEnum):
    PREFIX_LIMIT = "prefix-limit"
    PRUNE_MIDDLE = "prune-middle"


@dataclass(frozen=True)
class Context:
    max_tokens: int
    policy: ContextPolicy = ContextPolicy.PREFIX_LIMIT


@dataclass(frozen=True)
class TokenWindow:
    ids: list[int]
    positions: list[int]
    source_positions: list[int]


def prune_middle(
    ids: list[int], positions: list[int], cue_tokens: int, marker: list[int], max_tokens: int
) -> Iterator[TokenWindow]:
    """Yield one bounded input for each run of decisions sharing the same omission.

    Choose each omission at the first decision that would overflow. Reserve a
    quarter of the available history budget for later decisions in that window.
    """
    first_pruned = bisect_left(positions, max_tokens)
    if first_pruned == 0:
        raise ValueError("the goal and initial task context exceed the context cap")
    yield TokenWindow(ids[: positions[first_pruned - 1] + 1], positions[:first_pruned], positions[:first_pruned])
    if first_pruned == len(positions):
        return

    turn_starts = [position + 1 - cue_tokens for position in positions]
    protected = ids[: turn_starts[0]] + marker
    history_budget = max_tokens - len(protected)
    if history_budget <= cue_tokens:
        raise ValueError("the goal, initial task context, and omission marker leave no room for history")

    first = first_pruned
    while first < len(positions):
        first_end = positions[first] + 1
        earliest = first_end - history_budget
        target = earliest + (history_budget - cue_tokens) // 4
        start_turn = bisect_left(turn_starts, target, 1, first)
        if start_turn < first:
            tail_start = turn_starts[start_turn]
        elif turn_starts[first - 1] >= earliest:
            tail_start = turn_starts[first - 1]
        else:
            # A single action and its observation may exceed the history budget.
            tail_start = target

        shift = len(protected) - tail_start
        end = bisect_left(positions, max_tokens - shift, first)
        source_positions = positions[first:end]
        yield TokenWindow(
            protected + ids[tail_start : source_positions[-1] + 1],
            [position + shift for position in source_positions],
            source_positions,
        )
        first = end
