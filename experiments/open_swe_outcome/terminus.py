# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite Open-SWE outcome-ablation trajectories into Harbor's Terminus-2 JSON protocol.

Open-SWE-Traces come from three native-tool scaffolds: OpenHands, SWE-agent and mini-SWE-agent.
Marin evaluates on SWE-bench Verified with Harbor's Terminus-2 agent instead. Terminus-2 sends one
user prompt holding its instructions, the task and the terminal screen, and expects every reply to
be a JSON object whose ``commands`` are keystroke batches for a persistent tmux shell. Qwen3-8B
fine-tuned on the native-tool traces wrote malformed Terminus-2 replies at evaluation.

``terminus_row`` rewrites one arm row (``arms.chat_row`` output) into the conversation Terminus-2
would have produced had the teacher issued the same shell actions:

- The task message becomes Terminus-2's prompt around the issue text, which is all a SWE-bench
  Verified task gives the agent.
- Shell calls become keystrokes. Editor calls become the equivalent shell command: ``cat -n``,
  ``find``, a heredoc, or a short Python replacement script.
- Observations become the terminal output Terminus-2 shows: the echoed command after the shell
  prompt, then the output, truncated as Terminus-2 truncates it.
- ``submit``, ``finish`` and mini-SWE-agent's submission command become ``task_complete``, followed
  by Terminus-2's confirmation question and a confirming reply.
- OpenHands ``think`` calls join the next turn's reasoning. Repository paths under ``/workspace``
  become ``/testbed``, where SWE-bench Verified tasks start.

Rows with an operation that has no faithful shell equivalent raise ``Unconvertible``.

Tracked in marin-community/marin-logbooks#39.
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

HARBOR_TERMINUS_DIR = Path(__file__).resolve().parents[2] / "vendor/harbor/src/harbor/agents/terminus_2"
PROMPT_TEMPLATE = (HARBOR_TERMINUS_DIR / "templates/terminus-json-plain.txt").read_text()
# Terminus2._get_completion_confirmation_message for the JSON parser.
COMPLETION_QUESTION = (
    "Are you sure you want to mark the task as complete? This will trigger your solution to be graded "
    "and you won't be able to make any further corrections. "
    'If so, include "task_complete": true in your JSON response again.'
)
# Terminus2._limit_output_length's default.
OUTPUT_LIMIT_BYTES = 10_000
REPO_DIR = "/testbed"
# Blank rows below the prompt on Terminus-2's first screen in a SWE-bench Verified sandbox.
INITIAL_SCREEN_PADDING = 41
MINI_SWE_SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

QUICK_DURATION = 0.1
DEFAULT_DURATION = 1.0
SCRIPT_DURATION = 5.0
SLOW_DURATION = 30.0
QUICK_PROGRAMS = frozenset(
    "cat cd cp cut diff echo file find git grep head ls mkdir mv nl pwd rg rm sed sort stat tail touch "
    "tree uniq wc which".split()
)
SLOW_PATTERN = re.compile(
    r"\b(pytest|py\.test|tox|nox|pip3? install|poetry|npm|npx|yarn|pnpm|jest|mocha|go (?:test|build|run|vet)|"
    r"cargo|make|cmake|mvn|gradle|gradlew|tsc|bundle|rake|rspec|composer|phpunit|dotnet|sbt|ctest)\b"
)
SCRIPT_PATTERN = re.compile(r"\b(python3?|node|ruby|php|java|ts-node|deno|bun)\b")
# A command that changes shell state; mini-SWE-agent runs each command in a fresh shell.
STATEFUL_PATTERN = re.compile(r"(?:^|[;&|(\n]\s*)(?:cd|pushd|popd|export|source|unset|alias)\s")
LEADING_REPO_CD = re.compile(rf"^cd {re.escape(REPO_DIR)}\s*(?:&&|;)\s*")

OPENHANDS_METADATA = re.compile(
    r"^\[(?:The command completed with exit code \d+\.(?: CTRL\+C was sent\.)?|Current working directory: .*|"
    r"Python interpreter: .*|Command finished with exit code \d+|The command has no new output after \d+ seconds\..*)\]$"
)
OPENHANDS_PREVIOUS_OUTPUT = "[Below is the output of the previous command.]\n"
OPENHANDS_STILL_RUNNING = re.compile(r"^\[The command has no new output after (\d+) seconds\.", re.MULTILINE)
OPENHANDS_NOT_EXECUTED = "is NOT executed"
OPENHANDS_NOTHING_RUNNING = "No previous running command to interact with"
INTERRUPT = "C-c"
VIEW_RANGE_TOO_LONG = re.compile(r"Invalid `view_range`.*number of lines in the file: `(\d+)`", re.DOTALL)
VIEW_MISSING_PATH = re.compile(r"The path .* does not exist")
SWE_AGENT_PREFIX = "OBSERVATION:\n"
SWE_AGENT_SILENT = "Your command ran successfully and did not produce any output."
FILE_VIEW_HEADER = re.compile(r"^Here's the result of running `cat -n` on .*?:\n")
DIR_VIEW_HEADER = re.compile(r"^Here's the files and directories up to 2 levels deep in .*?, excluding hidden items:\n")
DIR_VIEW_NOTE = re.compile(
    r"\n+(?:\d+ hidden files/directories in this directory are excluded\..*|<NOTE>.*)$", re.DOTALL
)
EDIT_SUCCESS = "has been edited."
EDIT_NOT_FOUND = "did not appear verbatim"
EDIT_MULTIPLE = re.compile(r"Multiple occurrences of old_str .* in lines (\[[\d, ]*\])", re.DOTALL)
CREATE_SUCCESS = "File created successfully"
INSTRUCTION_TAGS = ("pr_description", "issue_description")
MINI_SWE_INSTRUCTION_PREFIX = "Consider the following PR description:\n"
UPLOADED_FILES = re.compile(r"<uploaded_files>\n(.*?)\n</uploaded_files>")


class Scaffold(StrEnum):
    OPENHANDS = "openhands"
    SWE_AGENT = "sweagent"
    MINI_SWE_AGENT = "minisweagent"


class Unconvertible(ValueError):
    """The trajectory has an operation with no faithful Terminus-2 equivalent."""


@dataclass(frozen=True)
class Action:
    """One keystroke batch and the terminal text it produced.

    ``echo`` is false for keys the shell does not print as typed: an interrupt, or an empty poll
    of a running command. ``running`` marks a command still running when Terminus-2 captured the
    screen, so no new prompt follows its output.
    """

    keystrokes: str
    duration: float
    output: str
    echo: bool = True
    running: bool = False


@dataclass
class Turn:
    reasoning: str
    analysis: str
    actions: list[Action] = field(default_factory=list)
    complete: bool = False


def scaffold_of(partition: str) -> Scaffold:
    for scaffold in Scaffold:
        if f"/{scaffold.value}/" in partition:
            return scaffold
    raise Unconvertible(f"unknown scaffold in partition {partition}")


def shell_prompt(trajectory_id: str) -> str:
    """The SWE-bench sandbox prompt, with a container name fixed per trajectory."""
    host = uuid.uuid5(uuid.NAMESPACE_URL, trajectory_id)
    return f"(testbed) root@{host}:{REPO_DIR}# "


def limit_output_length(output: str, max_bytes: int = OUTPUT_LIMIT_BYTES) -> str:
    """Terminus2._limit_output_length: keep the first and last halves of over-long output."""
    data = output.encode("utf-8")
    if len(data) <= max_bytes:
        return output
    half = max_bytes // 2
    first = data[:half].decode("utf-8", errors="ignore")
    last = data[-half:].decode("utf-8", errors="ignore")
    omitted = len(data) - len(first.encode("utf-8")) - len(last.encode("utf-8"))
    return f"{first}\n[... output limited to {max_bytes} bytes; {omitted} interior bytes omitted ...]\n{last}"


def instruction_text(task: str, scaffold: Scaffold) -> str:
    """The issue text inside the scaffold's task message."""
    for tag in INSTRUCTION_TAGS:
        match = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", task, re.DOTALL)
        if match:
            text = match.group(1)
            if scaffold is Scaffold.MINI_SWE_AGENT:
                text = text.removeprefix(MINI_SWE_INSTRUCTION_PREFIX)
            return text.strip("\n")
    raise Unconvertible("task message has no issue description")


def command_duration(command: str) -> float:
    if SLOW_PATTERN.search(command):
        return SLOW_DURATION
    if SCRIPT_PATTERN.search(command):
        return SCRIPT_DURATION
    programs = {segment.split()[0] for segment in re.split(r"&&|\|\||[;|\n]", command) if segment.split()}
    return QUICK_DURATION if programs <= QUICK_PROGRAMS else DEFAULT_DURATION


def shell_action(command: str, output: str) -> Action:
    command = command.strip("\n")
    return Action(keystrokes=command + "\n", duration=command_duration(command), output=output)


def isolated(command: str) -> str:
    """Run a mini-SWE-agent command in a subshell so its directory and variables do not persist."""
    command = command.strip("\n")
    if not STATEFUL_PATTERN.search(LEADING_REPO_CD.sub("", command, count=1)):
        return command
    return f"(\n{command}\n)"


def heredoc_delimiter(*texts: str) -> str:
    delimiter = "EOF"
    while any(delimiter in text.split("\n") for text in texts):
        delimiter += "_"
    return delimiter


def python_literal(text: str) -> str:
    """A raw triple-quoted literal when it reads back exactly; ``repr`` otherwise."""
    for quote in ("'", '"'):
        if quote * 3 not in text and not text.endswith((quote, "\\")):
            return f"r{quote * 3}{text}{quote * 3}"
    return repr(text)


def replace_script(path: str, old: str, new: str) -> str:
    body = "\n".join(
        (
            "from pathlib import Path",
            f"path = Path({path!r})",
            "text = path.read_text()",
            f"old = {python_literal(old)}",
            f"new = {python_literal(new)}",
            "count = text.count(old)",
            "if count == 1:",
            "    path.write_text(text.replace(old, new))",
            "else:",
            "    print(f'Error: the old text occurs {count} times in {path}; no edit made')",
        )
    )
    delimiter = heredoc_delimiter(body)
    return f"python3 - <<'{delimiter}'\n{body}\n{delimiter}"


def replace_failure(path: str, count: int) -> str:
    return f"Error: the old text occurs {count} times in {path}; no edit made\n"


def editor_action(arguments: dict, observation: str) -> Action:
    command = arguments.get("command")
    path = arguments.get("path")
    if not isinstance(path, str):
        raise Unconvertible("editor call has no path")
    if command == "view":
        if DIR_VIEW_HEADER.match(observation):
            listing = DIR_VIEW_NOTE.sub("", DIR_VIEW_HEADER.sub("", observation, count=1)).rstrip("\n")
            return shell_action(f"find {path} -maxdepth 2 -not -path '*/.*'", listing + "\n")
        too_long = VIEW_RANGE_TOO_LONG.search(observation)
        if too_long:
            # The editor rejects a range past the end and reports the file's length instead.
            return shell_action(f"wc -l {path}", f"{too_long.group(1)} {path}\n")
        if VIEW_MISSING_PATH.search(observation):
            return shell_action(f"ls {path}", f"ls: cannot access '{path}': No such file or directory\n")
        if not FILE_VIEW_HEADER.match(observation):
            raise Unconvertible("editor view failed")
        listing = FILE_VIEW_HEADER.sub("", observation, count=1).rstrip("\n") + "\n"
        view_range = arguments.get("view_range")
        if not view_range:
            return shell_action(f"cat -n {path}", listing)
        start, end = view_range
        stop = "$" if end == -1 else str(end)
        return shell_action(f"nl -ba {path} | sed -n '{start},{stop}p'", listing)
    if command == "create":
        if CREATE_SUCCESS not in observation:
            raise Unconvertible("editor create failed")
        text = arguments.get("file_text")
        if not isinstance(text, str):
            raise Unconvertible("editor create has no file_text")
        delimiter = heredoc_delimiter(text)
        return shell_action(f"cat > {path} <<'{delimiter}'\n{text.rstrip(chr(10))}\n{delimiter}", "")
    if command == "str_replace":
        old, new = arguments.get("old_str"), arguments.get("new_str", "")
        if not isinstance(old, str) or not isinstance(new, str):
            raise Unconvertible("editor str_replace has no old_str")
        script = replace_script(path, old, new)
        if EDIT_SUCCESS in observation:
            return shell_action(script, "")
        if EDIT_NOT_FOUND in observation:
            return shell_action(script, replace_failure(path, 0))
        multiple = EDIT_MULTIPLE.search(observation)
        if multiple:
            return shell_action(script, replace_failure(path, len(json.loads(multiple.group(1)))))
        raise Unconvertible("editor str_replace result not recognized")
    raise Unconvertible(f"editor command {command!r}")


def tool_output(observation: str, scaffold: Scaffold) -> str:
    """The text a terminal would show for one native tool result."""
    if scaffold is Scaffold.MINI_SWE_AGENT:
        try:
            result = json.loads(observation)
        except json.JSONDecodeError:
            return observation
        return result["output"] if isinstance(result, dict) and "output" in result else observation
    if scaffold is Scaffold.SWE_AGENT:
        text = observation.removeprefix(SWE_AGENT_PREFIX)
        return "" if text.strip() == SWE_AGENT_SILENT else text
    lines = observation.removeprefix(OPENHANDS_PREVIOUS_OUTPUT).split("\n")
    while lines and OPENHANDS_METADATA.match(lines[-1]):
        lines.pop()
    return "\n".join(lines)


def openhands_shell_action(arguments: dict, observation: str) -> Action:
    """An OpenHands shell call, including interrupts and polls of a command that is still running."""
    command = arguments["command"]
    if OPENHANDS_NOT_EXECUTED in observation:
        raise Unconvertible("shell command sent while another was running")
    output = tool_output(observation, Scaffold.OPENHANDS)
    still_running = OPENHANDS_STILL_RUNNING.search(observation)
    interactive = str(arguments.get("is_input", "false")).lower() == "true" or not command.strip()
    if not interactive:
        action = shell_action(command, output)
        if still_running:
            return Action(action.keystrokes, float(still_running.group(1)), output, running=True)
        return action
    if command == INTERRUPT:
        interrupted = "" if OPENHANDS_NOTHING_RUNNING in observation else output
        text = interrupted if not interrupted or interrupted.endswith("\n") else interrupted + "\n"
        return Action(INTERRUPT, QUICK_DURATION, text + "^C\n", echo=False)
    if command.strip():
        raise Unconvertible("interactive shell input")
    wait = (
        float(still_running.group(1)) if still_running else min(float(arguments.get("timeout") or SLOW_DURATION), 60.0)
    )
    return Action("", wait, output, echo=False, running=bool(still_running))


def call_arguments(call: dict) -> tuple[str, dict]:
    function = call["function"]
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments) if arguments.strip() else {}
    return function["name"], arguments


def converted_turns(messages: list[dict], scaffold: Scaffold) -> list[Turn]:
    """One Terminus-2 turn per native assistant turn, with OpenHands ``think`` calls folded forward."""
    turns: list[Turn] = []
    carried: list[str] = []
    index = 2
    while index < len(messages):
        message = messages[index]
        if message["role"] != "assistant":
            raise Unconvertible(f"unexpected {message['role']} message mid-trajectory")
        calls = message.get("tool_calls") or []
        observations = []
        for result in messages[index + 1 : index + 1 + len(calls)]:
            if result["role"] != "tool":
                break
            observations.append(result.get("content") or "")
        index += 1 + len(observations)
        if len(observations) < len(calls) and index < len(messages):
            raise Unconvertible("tool call without its result")
        reasoning = "\n\n".join(part for part in (*carried, (message.get("reasoning_content") or "").strip()) if part)
        content = (message.get("content") or "").strip()
        turn = Turn(reasoning=reasoning, analysis=content)
        thoughts = []
        for position, call in enumerate(calls):
            name, arguments = call_arguments(call)
            observation = observations[position] if position < len(observations) else ""
            if name == "think":
                thoughts.append(str(arguments.get("thought", "")).strip())
            elif name in ("submit", "finish"):
                turn.complete = True
                summary = str(arguments.get("message", "")).strip()
                turn.analysis = "\n\n".join(part for part in (content, summary) if part)
            elif name in ("bash", "execute_bash"):
                command = arguments.get("command")
                if not isinstance(command, str):
                    raise Unconvertible("shell call has no command")
                if MINI_SWE_SUBMIT_MARKER in command:
                    turn.complete = True
                elif scaffold is Scaffold.OPENHANDS:
                    turn.actions.append(openhands_shell_action(arguments, observation))
                elif scaffold is Scaffold.MINI_SWE_AGENT:
                    turn.actions.append(shell_action(isolated(command), tool_output(observation, scaffold)))
                else:
                    turn.actions.append(shell_action(command, tool_output(observation, scaffold)))
            elif name == "str_replace_editor":
                turn.actions.append(editor_action(arguments, tool_output(observation, scaffold)))
            else:
                raise Unconvertible(f"tool {name!r}")
        if thoughts and not turn.actions and not turn.complete:
            carried = [part for part in (reasoning, content, *thoughts) if part]
            continue
        if thoughts:
            turn.reasoning = "\n\n".join(part for part in (turn.reasoning, *thoughts) if part)
        carried = []
        turns.append(turn)
    if not turns:
        raise Unconvertible("no assistant turns")
    return turns


def reply_json(turn: Turn) -> str:
    reply = {
        "analysis": turn.analysis,
        "plan": "",
        "commands": [{"keystrokes": action.keystrokes, "duration": action.duration} for action in turn.actions],
        "task_complete": turn.complete,
    }
    return json.dumps(reply, indent=2, ensure_ascii=False)


def echoed(prompt: str, action: Action) -> str:
    """The terminal text for one command: the prompt, the typed lines and the output."""
    text = action.output
    if action.echo:
        lines = action.keystrokes.rstrip("\n").split("\n")
        text = prompt + lines[0] + "\n" + "".join(f"> {line}\n" for line in lines[1:]) + text
    return text if not text or text.endswith("\n") else text + "\n"


def terminal_output(prompt: str, actions: list[Action], first: bool) -> str:
    """Terminus-2's incremental output after a command batch."""
    if not actions:
        return f"Current Terminal Screen:\n{prompt.rstrip()}\n"
    body = "".join(echoed(prompt, action) for action in actions)
    final_prompt = "" if actions[-1].running else prompt.rstrip() + "\n"
    return "New Terminal Output:\n" + ("" if first else "\n") + body + final_prompt


def terminus_messages(instruction: str, turns: list[Turn], prompt: str) -> list[dict]:
    initial_screen = "Current Terminal Screen:\n" + prompt.rstrip() + "\n" * INITIAL_SCREEN_PADDING
    messages = [
        {"role": "user", "content": PROMPT_TEMPLATE.format(instruction=instruction, terminal_state=initial_screen)}
    ]
    pending_completion = False
    first_output = True
    for position, turn in enumerate(turns):
        messages.append({"role": "assistant", "content": reply_json(turn), "reasoning_content": turn.reasoning})
        if turn.complete and pending_completion:
            return messages
        output = terminal_output(prompt, turn.actions, first_output)
        first_output = first_output and not turn.actions
        if turn.complete:
            pending_completion = True
            observation = f"Current terminal state:\n{output}\n\n{COMPLETION_QUESTION}"
        else:
            pending_completion = False
            observation = limit_output_length(output)
        if position < len(turns) - 1 or turn.complete:
            messages.append({"role": "user", "content": observation})
    if pending_completion:
        messages.append(
            {"role": "assistant", "content": reply_json(Turn("", "", complete=True)), "reasoning_content": ""}
        )
    return messages


def repo_path_rewritten(messages: list[dict]) -> list[dict]:
    """Move an OpenHands ``/workspace/<repo>`` checkout to ``/testbed``, where Verified tasks start."""
    match = UPLOADED_FILES.search(messages[1]["content"])
    if not match or match.group(1) == REPO_DIR:
        return messages
    repo = match.group(1).rstrip("/")
    return json.loads(json.dumps(messages, ensure_ascii=False).replace(repo, REPO_DIR))


def terminus_row(row: dict) -> dict:
    """The Terminus-2 rendering of one arm row; raises ``Unconvertible``."""
    scaffold = scaffold_of(row["partition"])
    messages = repo_path_rewritten(row["messages"])
    if [m["role"] for m in messages[:2]] != ["system", "user"]:
        raise Unconvertible("trajectory does not start with system and task messages")
    instruction = instruction_text(messages[1]["content"], scaffold)
    turns = converted_turns(messages, scaffold)
    return {
        "messages": terminus_messages(instruction, turns, shell_prompt(row["trajectory_id"])),
        "chat_template_kwargs": {},
        "trajectory_id": row["trajectory_id"],
        "partition": row["partition"],
        "resolved": row["resolved"],
    }
