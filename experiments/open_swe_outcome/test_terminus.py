# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess

import pytest

from experiments.open_swe_outcome.terminus import (
    COMPLETION_QUESTION,
    HARBOR_TERMINUS_DIR,
    OUTPUT_LIMIT_BYTES,
    Unconvertible,
    terminus_row,
)

ISSUE = "zpath should return the uncompressed path when only it exists"


def _harbor_parser():
    """Harbor's own Terminus-2 JSON parser, which grades every reply at evaluation."""
    spec = importlib.util.spec_from_file_location(
        "terminus_parser", HARBOR_TERMINUS_DIR / "terminus_json_plain_parser.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TerminusJSONPlainParser()


def _call(name: str, **arguments) -> dict:
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _assistant(*calls: dict, content: str = "", reasoning: str = "") -> dict:
    return {"role": "assistant", "content": content, "reasoning_content": reasoning, "tool_calls": list(calls)}


def _tool(content: str) -> dict:
    return {"role": "tool", "content": content}


def _row(partition: str, task: str, trajectory: list[dict]) -> dict:
    return {
        "messages": [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": task}, *trajectory],
        "chat_template_kwargs": {"tools": []},
        "trajectory_id": f"{partition}-0",
        "partition": f"open_swe_traces/v1_0/{partition}/teacher/swe_rebench_v2",
        "resolved": 1,
    }


def _mini_swe_row() -> dict:
    task = (
        f"<pr_description>\nConsider the following PR description:\n{ISSUE}\n</pr_description>\n\n"
        "<instructions>\nSubmit with echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.\n</instructions>"
    )
    return _row(
        "minisweagent",
        task,
        [
            _assistant(_call("bash", command="cd /testbed && grep -rn zpath ."), reasoning="Find zpath."),
            _tool(json.dumps({"returncode": 0, "output": "monty/os/path.py:17:def zpath(filename):\n"})),
            _assistant(
                _call("bash", command="cd /testbed && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt")
            ),
        ],
    )


def _swe_agent_row() -> dict:
    task = (
        "<uploaded_files>\n/testbed\n</uploaded_files>\nConsider the following PR description:\n\n"
        f"<pr_description>\n{ISSUE}\n</pr_description>\n\nMake the <pr_description> pass."
    )
    return _row(
        "sweagent",
        task,
        [
            _assistant(_call("str_replace_editor", command="view", path="/testbed/path.py"), content="Read it."),
            _tool("OBSERVATION:\nHere's the result of running `cat -n` on /testbed/path.py:\n     1\tdef zpath():\n\n"),
            _assistant(
                _call("str_replace_editor", command="str_replace", path="/testbed/path.py", old_str="a", new_str="b")
            ),
            _tool("OBSERVATION:\nThe file /testbed/path.py has been edited. Here's the result of running `cat -n`:\n"),
            _assistant(_call("submit")),
            _tool("OBSERVATION:\nThank you for your work on this issue. Please review your changes."),
            _assistant(_call("submit")),
        ],
    )


def _openhands_row() -> dict:
    repo = "/workspace/materialsvirtuallab__monty__1.0"
    task = (
        f"<uploaded_files>\n{repo}\n</uploaded_files>\nConsider the following issue description:\n\n"
        f"<issue_description>\n{ISSUE}\n</issue_description>\n\nFix it."
    )
    return _row(
        "openhands",
        task,
        [
            _assistant(_call("execute_bash", command=f"cd {repo} && python -m pytest -q")),
            _tool(f"1 passed\n[The command completed with exit code 0.]\n[Current working directory: {repo}]"),
            _assistant(_call("think", thought="The fix belongs in zpath.")),
            _tool("Your thought has been logged."),
            _assistant(_call("finish", message="Done.")),
        ],
    )


@pytest.mark.parametrize("make_row", [_mini_swe_row, _swe_agent_row, _openhands_row])
def test_converted_trajectory_follows_the_terminus_protocol(make_row):
    messages = terminus_row(make_row())["messages"]
    parser = _harbor_parser()

    assert [m["role"] for m in messages] == ["user", "assistant"] * (len(messages) // 2)
    assert ISSUE in messages[0]["content"] and "COMPLETE_TASK_AND_SUBMIT" not in messages[0]["content"]
    results = [parser.parse_response(m["content"]) for m in messages if m["role"] == "assistant"]
    assert [(r.error, r.warning) for r in results] == [("", "")] * len(results)
    # Terminus-2 grades only after the agent marks the task complete twice in a row.
    assert [r.is_task_complete for r in results[-2:]] == [True, True]
    assert messages[-2]["content"].endswith(COMPLETION_QUESTION)


def test_openhands_think_joins_the_next_reply_and_paths_move_to_testbed():
    messages = terminus_row(_openhands_row())["messages"]
    replies = [m for m in messages if m["role"] == "assistant"]

    assert "/workspace" not in json.dumps(messages)
    assert "cd /testbed && python -m pytest -q" in replies[0]["content"]
    assert "The fix belongs in zpath." in replies[1]["reasoning_content"]
    assert "[The command completed" not in messages[2]["content"]


def test_editor_replacement_command_edits_the_file(tmp_path):
    source = tmp_path / "path.py"
    source.write_text("def zpath():\n    return '''x'''\\\n")
    old, new = "return '''x'''\\\n", 'return "y"  # \\n stays literal\n'
    row = _row(
        "sweagent",
        f"<pr_description>\n{ISSUE}\n</pr_description>",
        [
            _assistant(_call("str_replace_editor", command="str_replace", path=str(source), old_str=old, new_str=new)),
            _tool(f"OBSERVATION:\nThe file {source} has been edited."),
        ],
    )
    reply = json.loads(terminus_row(row)["messages"][1]["content"])

    subprocess.run(["bash", "-c", reply["commands"][0]["keystrokes"]], check=True)

    assert source.read_text() == 'def zpath():\n    return "y"  # \\n stays literal\n'


def test_mini_swe_directory_changes_do_not_persist_in_the_terminal(tmp_path):
    row = _mini_swe_row()
    row["messages"][2] = _assistant(_call("bash", command="cd sub && pwd"))
    row["messages"][3] = _tool(json.dumps({"returncode": 0, "output": "sub\n"}))
    (tmp_path / "sub").mkdir()
    keystrokes = json.loads(terminus_row(row)["messages"][1]["content"])["commands"][0]["keystrokes"]

    result = subprocess.run(
        ["bash", "-c", keystrokes + "pwd\n"], cwd=tmp_path, capture_output=True, text=True, check=True
    )

    assert result.stdout.splitlines() == [str(tmp_path / "sub"), str(tmp_path)]


def test_long_output_is_truncated_as_terminus_truncates_it():
    row = _mini_swe_row()
    row["messages"][3] = _tool(json.dumps({"returncode": 0, "output": "x" * 50_000}))
    observation = terminus_row(row)["messages"][2]["content"]

    assert f"[... output limited to {OUTPUT_LIMIT_BYTES} bytes;" in observation
    assert len(observation.encode()) < OUTPUT_LIMIT_BYTES + 200


def test_undo_edit_is_rejected():
    row = _swe_agent_row()
    row["messages"][4] = _assistant(_call("str_replace_editor", command="undo_edit", path="/testbed/path.py"))

    with pytest.raises(Unconvertible):
        terminus_row(row)
