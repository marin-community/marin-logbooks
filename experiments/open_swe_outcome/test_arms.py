# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import random
from collections import Counter

from experiments.open_swe_outcome.arms import (
    ARM_ALL,
    ARM_RANDOM_ALL,
    ARM_RANDOM_FIRST,
    ARM_SOLVED_ALL,
    ARM_SOLVED_FIRST,
    IndexRow,
    arm_members,
    chat_row,
)

GRADED = "open_swe_traces/v1_0/sweagent/minimax_m25/swe_rebench_v2"
UNGRADED = "open_swe_traces/v1_1/openhands/qwen36_27b/scale_swe"


def _kept_rows() -> list[IndexRow]:
    rng = random.Random(0)
    rows = [
        IndexRow(f"graded-{i}", GRADED, rng.choice((1, 0, 0, -1)), rng.randint(50_000, 400_000)) for i in range(2_000)
    ]
    rows += [IndexRow(f"ungraded-{i}", UNGRADED, -1, rng.randint(50_000, 400_000)) for i in range(1_000)]
    return rows


def _chars(rows: list[IndexRow]) -> int:
    return sum(row.num_chars for row in rows)


def test_random_arm_matches_solved_arm_in_partition_and_volume_but_not_outcome():
    kept = _kept_rows()
    members = arm_members(kept, first_rung_rows=100)

    solved, matched = members[ARM_SOLVED_FIRST], members[ARM_RANDOM_FIRST]
    assert {row.resolved for row in solved} == {1}
    assert len(solved) == 100
    # The solved arm never draws from the ungraded partition, so neither does its control.
    assert {row.partition for row in matched} == {GRADED}
    assert Counter(row.resolved for row in matched).keys() == {1, 0, -1}
    largest_row = max(row.num_chars for row in kept)
    assert 0 <= _chars(matched) - _chars(solved) < largest_row


def test_first_rung_is_nested_in_the_full_arms_and_selection_is_order_independent():
    kept = _kept_rows()
    members = arm_members(kept, first_rung_rows=100)
    ids = {arm: {row.trajectory_id for row in rows} for arm, rows in members.items()}

    assert ids[ARM_SOLVED_FIRST] < ids[ARM_SOLVED_ALL]
    assert ids[ARM_RANDOM_FIRST] < ids[ARM_RANDOM_ALL]
    assert ids[ARM_ALL] == {row.trajectory_id for row in kept}

    reordered = arm_members(list(reversed(kept)), first_rung_rows=100)
    assert {arm: {row.trajectory_id for row in rows} for arm, rows in reordered.items()} == ids


def test_chat_row_links_parallel_tool_results_and_drops_null_fields():
    tool = {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}
    call = {"type": "function", "function": {"name": "bash", "arguments": "{}"}}
    row = {
        "trajectory_id": "t-1",
        "resolved": 1,
        "tools": [json.dumps(tool)],
        "messages": [
            {"role": "system", "content": "sys", "tool_calls": None, "reasoning_content": None},
            {"role": "user", "content": "task", "tool_calls": None, "reasoning_content": None},
            {"role": "assistant", "content": None, "tool_calls": [call, call], "reasoning_content": "think"},
            {"role": "tool", "content": "first", "tool_calls": None, "reasoning_content": None},
            {"role": "tool", "content": "second", "tool_calls": None, "reasoning_content": None},
        ],
    }

    out = chat_row(row, GRADED)

    assistant, first, second = out["messages"][2:]
    assert [call["id"] for call in assistant["tool_calls"]] == [first["tool_call_id"], second["tool_call_id"]]
    assert "content" not in assistant
    assert all(value is not None for message in out["messages"] for value in message.values())
    assert out["chat_template_kwargs"] == {"tools": [tool]}
