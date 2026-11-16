# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise bounded judge retry and preservation through the real OpenAI client."""

import json
from pathlib import Path

import httpx
import pytest
from grade_with_retry import grade
from llm_judge import JudgeConfig, JudgeResponseError, parse_rubric
from openai import OpenAI

RUBRIC = Path(__file__).resolve().parent.parent / "pilot/remaining-tasks/da-20-3/tests/rubric.txt"


@pytest.mark.parametrize("replies", ["valid", "retry", "invalid", "empty", "missing_directory"])
def test_grading_recovery_preserves_reports_and_missing_grades(tmp_path, replies):
    submission = tmp_path / "submission"
    if replies != "missing_directory":
        submission.mkdir()
    if replies not in ("empty", "missing_directory"):
        (submission / "answer.txt").write_text("Frozen agent answer")
    levels = parse_rubric(RUBRIC.read_text())
    complete = json.dumps(
        {
            "criteria": {name: {"level": "A", "reason": "Fixture rating"} for name in levels},
            "overall_reasoning": "Fixture",
        }
    )
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        content = "{}" if replies == "invalid" or (replies == "retry" and len(calls) == 1) else complete
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "created": 0,
                "model": "fixture",
                "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            },
        )

    output = tmp_path / "grade"
    with OpenAI(
        api_key="fixture",
        base_url="https://judge.example/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        config = JudgeConfig(model="fixture", base_url="https://judge.example/v1")
        if replies == "invalid":
            with pytest.raises(JudgeResponseError):
                grade(RUBRIC, submission, output, config, client)
            assert len(calls) == 2
            assert not (output / "reward.json").exists()
            assert (output / "judge-attempts/1/error.json").exists()
        elif replies == "missing_directory":
            with pytest.raises(FileNotFoundError):
                grade(RUBRIC, submission, output, config, client)
            assert not calls
            assert not (output / "reward.json").exists()
        else:
            grade(RUBRIC, submission, output, config, client)
            assert len(calls) == {"valid": 1, "retry": 2, "empty": 0}[replies]
            evaluation = json.loads((output / "evaluation.json").read_text())
            assert evaluation["total_score"] == (0 if replies == "empty" else 100)
            if replies == "retry":
                assert (output / "judge-attempts/0/error.json").exists()
                assert calls[0] == calls[1]
        if replies not in ("empty", "missing_directory"):
            assert (output / "answer.txt").read_bytes() == (submission / "answer.txt").read_bytes()
