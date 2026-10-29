# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy

from experiments.post_training.identity_path_b.replay_capture import pilot_passes


def plain_protocol_pilot():
    rows = []
    for family in ("math", "code", "chat", "call_after_result", "first_call", "reply"):
        for thinking in (False, True):
            tool = family in {"call_after_result", "first_call", "reply"}
            rows.append(
                {
                    "candidate": {
                        "family": "tool" if tool else family,
                        "enable_thinking": thinking,
                        "teacher_system": "",
                        "teacher": "plain",
                        "tools": [{"function": {"name": "fixture_fn"}}] if tool else [],
                    },
                    "raw_token_ids": [128005, 42, 128009] if tool else [42, 128009],
                    "roundtrip": {
                        "ok": True,
                        "content": (
                            '<tool_call>{"name":"fixture_fn","arguments":{}}</tool_call>' if tool else "plain answer"
                        ),
                    },
                    "length_stop": False,
                }
            )
    return rows


def test_plain_live12_has_no_identity_teacher_or_name_first_requirement():
    rows = plain_protocol_pilot()
    assert pilot_passes(rows)
    missing_off_calls = copy.deepcopy(rows)
    for row in missing_off_calls:
        if not row["candidate"]["enable_thinking"]:
            row["raw_token_ids"] = [42, 128009]
    assert not pilot_passes(missing_off_calls)
    bad_fidelity = copy.deepcopy(rows)
    bad_fidelity[0]["roundtrip"]["ok"] = False
    assert not pilot_passes(bad_fidelity)
    bad_stop = copy.deepcopy(rows)
    bad_stop[0]["length_stop"] = True
    assert not pilot_passes(bad_stop)
