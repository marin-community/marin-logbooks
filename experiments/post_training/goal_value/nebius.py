# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build same-issue comparisons from pinned Nebius OpenHands attempts."""

import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from experiments.post_training.goal_value.data import Episode, stable_hash

DATASET = "nebius/SWE-rebench-openhands-trajectories"
REVISION = "35455389ab51bf5e2306bfd436ef72d0f98bf882"
DATA_FILE = "trajectories.parquet"
ACTOR = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
CUTOFFS = (0, 1, 2, 4, 8, 16)
SAMPLE_KEY = "nebius-openhands-mixed-79-v1"
GOAL = (
    "Submit a repository patch that resolves the issue. The SWE-rebench grader checks the patch with repository tests."
)


class InvalidTrace(ValueError):
    """The recorded history lacks a usable task or tool boundary."""


@dataclass(frozen=True)
class Scorable:
    episode: Episode
    cutoffs: dict[int, int]


def select_attempts(source: pq.ParquetFile, excluded: set[str], issue_limit: int) -> tuple[set[str], dict]:
    """Draw issues by a stable hash and retain their distinct submitted trajectories."""
    by_issue: dict[str, dict[str, int]] = defaultdict(dict)
    counts = Counter()
    for row in source.read(columns=["instance_id", "trajectory_id", "resolved", "exit_status"]).to_pylist():
        if row["exit_status"] != "submit":
            continue
        outcome = row["resolved"]
        if outcome not in (0, 1):
            counts["invalid_label_rows"] += 1
            continue
        issue = row["instance_id"]
        if issue.casefold() in excluded:
            counts["overlap_rows"] += 1
            continue
        attempt = row["trajectory_id"]
        if attempt in by_issue[issue]:
            if by_issue[issue][attempt] != outcome:
                raise ValueError("a duplicate trajectory has conflicting outcomes")
            counts["duplicate_rows"] += 1
        by_issue[issue][attempt] = outcome

    mixed = [issue for issue, attempts in by_issue.items() if set(attempts.values()) == {0, 1}]
    selected = sorted(mixed, key=lambda issue: (stable_hash(SAMPLE_KEY, issue), issue))[:issue_limit]
    selected_attempts = {attempt for issue in selected for attempt in by_issue[issue]}
    if len(selected_attempts) != sum(len(by_issue[issue]) for issue in selected):
        raise ValueError("a selected trajectory ID is shared by multiple issues")
    audit = {
        "dataset": DATASET,
        "revision": REVISION,
        "sample_key": SAMPLE_KEY,
        "eligible_mixed_issues": len(mixed),
        "selected_issues": len(selected),
        "selected_trajectories": len(selected_attempts),
        "selected_passes": sum(sum(by_issue[issue].values()) for issue in selected),
        "excluded_issue_ids": len(excluded),
        **counts,
    }
    return selected_attempts, audit


def selected_rows(source: pq.ParquetFile, attempt_ids: set[str]) -> Iterator[dict]:
    """Filter Arrow batches before copying selected histories into Python."""
    value_set = pa.array(sorted(attempt_ids))
    seen: set[str] = set()
    columns = ["trajectory_id", "instance_id", "repo", "trajectory", "resolved", "exit_status"]
    for batch in source.iter_batches(batch_size=192, columns=columns):
        mask = pc.is_in(batch.column("trajectory_id"), value_set=value_set)
        for row in batch.filter(mask).to_pylist():
            attempt = row["trajectory_id"]
            if attempt not in seen and row["exit_status"] == "submit" and row["resolved"] in (0, 1):
                seen.add(attempt)
                yield row
    if seen != attempt_ids:
        raise ValueError("selected trajectory coverage changed between reads")


def scorable(row: dict) -> Scorable:
    """Index actual assistant decisions after a completed batch of tool calls."""
    messages = []
    decisions = []
    cutoffs = {}
    initial_issue = ""
    completed = 0
    pending: set[str] = set()

    for raw in row["trajectory"]:
        role = raw["role"]
        content = raw["content"] or ""
        message = {"role": role, "content": content}
        if role == "user" and not initial_issue:
            initial_issue = content.strip()
        if role == "assistant":
            if not initial_issue:
                raise InvalidTrace("assistant decision precedes issue text")
            if pending:
                raise InvalidTrace("assistant acted before receiving all tool results")
            decision = len(messages)
            decisions.append(decision)
            if completed in CUTOFFS:
                cutoffs.setdefault(completed, decision)
            if completed == CUTOFFS[-1]:
                messages.append({"role": "assistant", "content": ""})
                break
            calls = []
            for call in raw["tool_calls"] or ():
                call_id = call["id"]
                if not call_id or call_id in pending:
                    raise InvalidTrace("missing or repeated tool call ID")
                function = call["function"]
                try:
                    arguments = json.loads(function["arguments"])
                except (ValueError, TypeError) as error:
                    raise InvalidTrace("invalid tool arguments") from error
                pending.add(call_id)
                calls.append({"tool_call_id": call_id, "function_name": function["name"], "arguments": arguments})
            if calls:
                message["tool_calls"] = calls
        elif role == "tool":
            call_id = raw["tool_call_id"]
            if call_id not in pending:
                raise InvalidTrace("tool response has no pending call")
            pending.remove(call_id)
            completed += not pending
        elif role not in ("system", "user"):
            raise InvalidTrace("unrecognized role")
        messages.append(message)

    if not initial_issue or not cutoffs:
        raise InvalidTrace("missing issue text or decision cue")
    last = max(cutoffs.values())
    episode = Episode(
        task_id=row["instance_id"],
        trajectory_id=row["trajectory_id"],
        family="repository-repair",
        repository=row["repo"],
        run_id=DATASET,
        run_timestamp="",
        source=DATASET,
        revision=REVISION,
        policy=ACTOR,
        checkpoint=ACTOR,
        checkpoint_revision=None,
        policy_config_digest="openhands-v0.54.0",
        goal=f"{GOAL}\nRepository: {row['repo']}\nIssue:\n{initial_issue}",
        messages=messages[: last + 1],
        predict_before=[decision for decision in decisions if decision <= last],
        outcome=row["resolved"],
    )
    return Scorable(episode, {position: cut for cut, position in cutoffs.items()})


def prepare(path: Path, excluded: set[str], issue_limit: int) -> tuple[list[Scorable], dict]:
    """Select and audit one deterministic, task-disjoint evaluation cohort."""
    source = pq.ParquetFile(path)
    attempt_ids, audit = select_attempts(source, excluded, issue_limit)
    cohort = []
    rejected = Counter()
    for row in selected_rows(source, attempt_ids):
        try:
            cohort.append(scorable(row))
        except InvalidTrace as error:
            rejected[str(error)] += 1
    audit["converted_trajectories"] = len(cohort)
    audit["rejected_trajectories"] = dict(rejected)
    audit["schema_coverage"] = {str(cut): sum(cut in case.cutoffs.values() for case in cohort) for cut in CUTOFFS}
    return cohort, audit
