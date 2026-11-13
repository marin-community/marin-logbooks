# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import random
from collections import Counter

from experiments.open_swe_outcome.arms import (
    ARM_ALL,
    ARM_RANDOM_ALL,
    ARM_SOLVED_ALL,
    IndexRow,
    arm_members,
    chat_row,
    random_arm,
    solved_arm,
)

GRADED = "open_swe_traces/v1_0/sweagent/minimax_m25/swe_rebench_v2"
UNGRADED = "open_swe_traces/v1_1/openhands/qwen36_27b/scale_swe"
MAX_TOKENS = 65_536
RUNGS = (100, 300)


def _kept_rows() -> list[IndexRow]:
    rng = random.Random(0)
    rows = [
        IndexRow(f"graded-{i}", GRADED, rng.choice((1, 0, 0, -1)), rng.randint(10_000, 120_000)) for i in range(4_000)
    ]
    rows += [IndexRow(f"ungraded-{i}", UNGRADED, -1, rng.randint(10_000, 120_000)) for i in range(1_000)]
    return rows


def _tokens(rows: list[IndexRow]) -> int:
    return sum(row.num_tokens for row in rows)


def test_random_arm_matches_solved_arm_in_partition_and_tokens_but_not_outcome():
    kept = _kept_rows()
    members = arm_members(kept, RUNGS, MAX_TOKENS)

    solved, matched = members[solved_arm(100)], members[random_arm(100)]
    assert {row.resolved for row in solved} == {1}
    assert len(solved) == 100
    # The solved arm never draws from the ungraded partition, so neither does its control.
    assert {row.partition for row in matched} == {GRADED}
    assert Counter(row.resolved for row in matched).keys() == {1, 0, -1}
    assert 0 <= _tokens(matched) - _tokens(solved) < MAX_TOKENS


def test_no_arm_holds_a_trajectory_longer_than_the_training_window():
    kept = _kept_rows()
    members = arm_members(kept, RUNGS, MAX_TOKENS)

    assert any(row.num_tokens > MAX_TOKENS for row in kept)
    assert all(row.num_tokens <= MAX_TOKENS for rows in members.values() for row in rows)
    assert {row.trajectory_id for row in members[ARM_ALL]} == {
        row.trajectory_id for row in kept if row.num_tokens <= MAX_TOKENS
    }


def test_rungs_are_nested_and_selection_is_order_independent():
    kept = _kept_rows()
    members = arm_members(kept, RUNGS, MAX_TOKENS)
    ids = {arm: {row.trajectory_id for row in rows} for arm, rows in members.items()}

    assert ids[solved_arm(100)] < ids[solved_arm(300)] < ids[ARM_SOLVED_ALL]
    assert ids[random_arm(100)] < ids[random_arm(300)] < ids[ARM_RANDOM_ALL]

    reordered = arm_members(list(reversed(kept)), RUNGS, MAX_TOKENS)
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
