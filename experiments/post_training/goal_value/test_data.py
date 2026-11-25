# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pytest

from experiments.post_training.goal_value.data import SUBMIT_COMMAND, RejectedEpisode, harbor_episode


def test_harbor_decisions_precede_actions_and_follow_observations():
    issue = "Please solve this issue: Repair the date parser."
    edit = {"tool_call_id": "first", "function_name": "bash", "arguments": {"command": "apply repair"}}
    submit = {"tool_call_id": "final", "function_name": "bash", "arguments": {"command": SUBMIT_COMMAND}}
    record = {
        "run_id": "run",
        "created_at": "2026-10-20T00:00:00Z",
        "results_path": "archive",
        "artifact_uri": "artifact",
        "task": "org__repo-42",
        "model": {"name": "checkpoint", "location": "model-location", "config_digest": "config"},
        "outcome": 0,
        "trajectory": {
            "schema_version": "ATIF-v1.2",
            "session_id": "session",
            "agent": {"name": "mini-swe-agent"},
            "steps": [
                {
                    "step_id": 1,
                    "source": "user",
                    "message": f"{issue}\n\n## Recommended Workflow\nRun on VM 42.",
                },
                {
                    "step_id": 2,
                    "source": "agent",
                    "tool_calls": [edit],
                    "observation": {"results": [{"content": "One local test passed."}]},
                },
                {"step_id": 3, "source": "agent", "tool_calls": [submit]},
            ],
            "final_metrics": {"verifier": "private result"},
        },
    }
    episode = harbor_episode(record)
    first, second = episode.predict_before
    assert [msg["role"] for msg in episode.messages[:first]] == ["user"]
    assert [msg["role"] for msg in episode.messages[:second]] == ["user", "assistant", "tool"]
    assert episode.messages[second - 1]["content"] == "One local test passed."
    assert episode.messages[second]["tool_calls"] == [submit]
    assert "Repair the date parser." in episode.goal
    assert "VM 42" not in episode.goal
    assert "VM 42" in episode.messages[0]["content"]
    assert "private result" not in json.dumps(episode.messages) + episode.goal
    assert harbor_episode(dict(record, outcome=1)).goal == episode.goal
    assert replace(episode, trajectory_id="another", policy="later").split == episode.split

    record["trajectory"]["steps"][1].pop("observation")
    with pytest.raises(RejectedEpisode, match="missing tool observation"):
        harbor_episode(record)
