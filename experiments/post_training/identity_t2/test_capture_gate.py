# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
from concurrent.futures import ThreadPoolExecutor

from experiments.post_training.identity_t2.capture import FidelityBudget, pilot_passes


def record(index, *, rejected=False):
    return {
        "index": index,
        "candidate": {"id": f"request-{index}"},
        "raw_token_ids": [366, 14506, 13735, 29, 128009],
        "roundtrip": {"ok": not rejected, "reason": "noncanonical_target_IDs" if rejected else None},
        "fidelity_rejected": rejected,
    }


def test_bulk_row_is_recorded_as_reject_without_repair_or_immediate_stop():
    budget = FidelityBudget()
    for index in range(5):
        assert not budget.observe(record(index))
    rejected = record(5, rejected=True)
    original = copy.deepcopy(rejected)
    assert not budget.observe(rejected)
    assert rejected == original
    assert budget.rejects == [{"index": 5, "id": "request-5", "reason": "noncanonical_target_IDs"}]
    for index in range(6, 100):
        assert not budget.observe(record(index))
    assert not budget.finish()


def test_exactly_five_percent_is_allowed():
    budget = FidelityBudget()
    for index in range(100):
        assert not budget.observe(record(index, rejected=5 <= index < 10))
    assert budget.attempted == 100 and len(budget.rejects) == 5
    assert budget.checkpoints == [{"attempted": 100, "fidelity_rejects": 5, "over_5_percent": False, "final": False}]
    assert not budget.finish()


def test_over_five_percent_stops_at_the_hundredth_completion():
    budget = FidelityBudget()
    for index in range(99):
        assert not budget.observe(record(index, rejected=5 <= index < 11))
    assert not budget.checkpoints
    assert budget.observe(record(99))
    assert budget.exceeded
    assert budget.checkpoints[-1]["attempted"] == 100
    assert budget.finish()


def test_final_partial_interval_cannot_hide_excess_faults():
    budget = FidelityBudget()
    for index in range(99):
        assert not budget.observe(record(index, rejected=5 <= index < 11))
    assert budget.finish()
    assert budget.checkpoints[-1]["final"]


def test_transport_or_protocol_fault_still_stops_immediately():
    budget = FidelityBudget()
    failed = record(0)
    failed["fatal"] = "transport_protocol_or_capture_failure"
    assert budget.observe(failed)
    assert budget.fatal_protocol and not budget.exceeded
    assert budget.attempted == 1 and not budget.checkpoints


def test_length_stop_keeps_its_separate_rejection_policy():
    budget = FidelityBudget()
    stopped = record(0)
    stopped["length_stop"] = True
    stopped["roundtrip"]["ok"] = False
    assert not budget.observe(stopped)
    assert not budget.rejects
    assert not budget.finish()


def test_completed_attempt_counter_and_checkpoints_are_thread_safe():
    budget = FidelityBudget()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda index: budget.observe(record(index, rejected=index > 5 and index % 25 == 0)), range(1000))
        )
    assert not any(results)
    assert budget.attempted == 1000 and len(budget.rejects) == 39
    assert [point["attempted"] for point in budget.checkpoints] == list(range(100, 1001, 100))
    assert not budget.finish()


def test_live_five_remains_fail_closed_even_though_bulk_can_drop_rows():
    tool = {"function": {"name": "example"}}
    rows = [
        {
            "candidate": {"group": "tools", "tools": [tool]},
            "raw_token_ids": [128005, 123, 128011, 128009],
            "roundtrip": {"ok": True, "content": '<tool_call>{"name":"example","arguments":{}}</tool_call>'},
        }
        for _ in range(4)
    ]
    rows.append(
        {
            "candidate": {"group": "identity_v2", "teacher": "style_d"},
            "roundtrip": {"ok": True, "content": "Marin Snowball."},
        }
    )
    assert pilot_passes(rows)
    rows[0]["roundtrip"]["ok"] = False
    rows[0]["fidelity_rejected"] = True
    assert not pilot_passes(rows)
