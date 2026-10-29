# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest

from experiments.post_training.identity_path_b import replay_capture
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
                        "tools": (
                            [
                                {
                                    "type": "function",
                                    "function": {"name": "fixture_fn", "description": "fixture", "parameters": {}},
                                }
                            ]
                            if tool
                            else []
                        ),
                    },
                    "http_status": 200,
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
    bad_fidelity[0]["fidelity_rejected"] = True
    assert pilot_passes(bad_fidelity)
    bad_stop = copy.deepcopy(rows)
    bad_stop[0]["length_stop"] = True
    assert pilot_passes(bad_stop)
    bad_prompt = copy.deepcopy(rows)
    bad_prompt[0]["fatal"] = "transport_protocol_or_capture_failure"
    assert not pilot_passes(bad_prompt)


class PrefixTokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return [1]


def capture_fixture(monkeypatch, rejects: set[int], prompt_mismatch: int | None):
    """A terminal raw-capture transport, not a native-tokenizer substitute."""
    templates = plain_protocol_pilot()

    def capture_one(index, candidate, **kwargs):
        out = kwargs["out"]
        record = copy.deepcopy(templates[index % 12])
        record.update(
            index=index, candidate=candidate, cpu_teacher_prompt_token_ids=[1], server_teacher_prompt_token_ids=[1]
        )
        if index in rejects:
            record.update(fidelity_rejected=True, roundtrip={"ok": False, "reason": "fixture_output_fidelity"})
        if index == prompt_mismatch:
            record.update(server_teacher_prompt_token_ids=[2], fatal="transport_protocol_or_capture_failure")
            record.pop("http_status")
        (out / f"{index:04d}.json").write_text(json.dumps(record) + "\n")
        return record

    monkeypatch.setattr(replay_capture.t2, "capture_one", capture_one)
    rows = []
    for index in range(300):
        row = copy.deepcopy(templates[index % 12]["candidate"])
        row.update(id=f"fixture-{index}", prompt=[{"role": "user", "content": "fixture"}])
        rows.append(row)
    return rows


def test_fixed_rate_cadence_allows_live_drops_but_stops_above_five_percent_and_drains_issued(monkeypatch, tmp_path):
    rows = capture_fixture(monkeypatch, set(range(5)) | set(range(110, 116)), None)
    with pytest.raises(RuntimeError, match="output-fidelity budget"):
        replay_capture.capture(
            rows,
            tokenizer=PrefixTokenizer(),
            endpoint="https://fixture/v1",
            model="fixture",
            output=tmp_path / "raw",
            artifact_root="s3://marin-rno2a/tmp/ttl=14d/fixture/",
        )
    report = json.loads((tmp_path / "raw/capture-receipt.json").read_bytes())
    assert report["live12_passes"]
    checks = report["output_fidelity_rate_checkpoints"]
    assert checks[0] == {"attempted": 100, "fidelity_rejects": 5, "over_5_percent": False, "final": False}
    assert checks[1] == {"attempted": 200, "fidelity_rejects": 11, "over_5_percent": True, "final": False}
    assert checks[-1]["final"] and checks[-1]["attempted"] == report["attempts"]
    assert 200 <= report["attempts"] <= 207
    assert report["requests_not_started"] == 300 - report["capture_records"]
    assert len(list((tmp_path / "raw").glob("[0-9][0-9][0-9][0-9].json"))) == report["capture_records"]
    raw = json.loads((tmp_path / "raw/0004.json").read_bytes())
    assert raw["raw_token_ids"] == [42, 128009] and raw["fidelity_rejected"]


def test_prompt_mismatch_stops_before_any_further_completion_attempt_and_keeps_raw(monkeypatch, tmp_path):
    rows = capture_fixture(monkeypatch, {4}, 6)
    with pytest.raises(RuntimeError, match="prompt/protocol/tool gate"):
        replay_capture.capture(
            rows,
            tokenizer=PrefixTokenizer(),
            endpoint="https://fixture/v1",
            model="fixture",
            output=tmp_path / "raw",
            artifact_root="s3://marin-rno2a/tmp/ttl=14d/fixture/",
        )
    report = json.loads((tmp_path / "raw/capture-receipt.json").read_bytes())
    assert report["capture_records"] == 7 and report["attempts"] == 6
    assert report["fatal_protocol_or_prompt"] and not report["live12_passes"]
    assert not (tmp_path / "raw/0007.json").exists()
    mismatch = json.loads((tmp_path / "raw/0006.json").read_bytes())
    assert mismatch["server_teacher_prompt_token_ids"] == [2] and "http_status" not in mismatch
