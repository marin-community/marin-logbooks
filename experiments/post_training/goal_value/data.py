# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert graded Harbor SWE-bench Verified ATIF sessions into value-model examples."""

import argparse
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

BENCHMARK_REVISION = "swebench-verified@1.0"
REPAIR_CRITERION = (
    "Submit a repository patch that resolves the issue. The SWE-bench Verified grader checks "
    "that the patch passes the tests for the reported issue and keeps the regression tests passing."
)
SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MINI_SWE_AGENT_WRAPPER = "\n## Recommended Workflow\n"


@dataclass(frozen=True)
class Episode:
    task_id: str
    trajectory_id: str
    family: str
    repository: str
    run_id: str
    run_timestamp: str
    source: str
    revision: str
    policy: str
    checkpoint: str
    checkpoint_revision: str | None
    policy_config_digest: str
    goal: str
    messages: list[dict]
    predict_before: list[int]
    outcome: int

    @property
    def split(self) -> str:
        """Keep every attempt at a task in one stable split."""
        return task_split(self.task_id)


class RejectedEpisode(ValueError):
    """A trajectory with incomplete or ambiguous decision history."""


def stable_hash(namespace: str, identity: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{namespace}\0{identity}".encode()).digest()[:8], "big")


def task_split(task_id: str) -> str:
    bucket = stable_hash("split", task_id) % 100
    if bucket < 70:
        return "train"
    return "validation" if bucket < 85 else "test"


def harbor_episode(record: dict) -> Episode:
    """Preserve the ATIF action and observation order, excluding trial metadata from the prefix."""
    trajectory = record["trajectory"]
    if trajectory["schema_version"] != "ATIF-v1.2" or trajectory["agent"]["name"] != "mini-swe-agent":
        raise RejectedEpisode("unsupported trajectory format or agent")
    if record["outcome"] not in (0, 1):
        raise RejectedEpisode("missing binary verdict")

    steps = trajectory["steps"]
    if [step["step_id"] for step in steps] != list(range(1, len(steps) + 1)):
        raise RejectedEpisode("noncontiguous steps")
    if not steps or steps[-1]["source"] != "agent":
        raise RejectedEpisode("missing final assistant action")
    final_calls = steps[-1].get("tool_calls") or []
    if steps[-1].get("observation") or not any(
        call["function_name"] == "bash" and SUBMIT_COMMAND in call["arguments"].get("command", "")
        for call in final_calls
    ):
        raise RejectedEpisode("no clean submission")

    messages = []
    positions = []
    issue = ""
    model_names = {step["model_name"] for step in steps if step.get("model_name")}
    if len(model_names) > 1:
        raise RejectedEpisode("multiple acting models")

    for index, step in enumerate(steps):
        source = step["source"]
        content = step.get("message") or ""
        if source in ("system", "user"):
            if positions:
                raise RejectedEpisode("unexpected system/user step after agent action")
            messages.append({"role": source, "content": content})
            if source == "user" and not issue:
                payload, separator, _ = content.rpartition(MINI_SWE_AGENT_WRAPPER)
                issue = (payload if separator else content).removeprefix("Please solve this issue: ").strip()
            continue
        if source != "agent":
            raise RejectedEpisode("unrecognized step source")

        reasoning = step.get("reasoning_content")
        calls = step.get("tool_calls") or []
        if not (content or reasoning or calls):
            raise RejectedEpisode("empty assistant action")
        positions.append(len(messages))
        message = {"role": "assistant", "content": content}
        if reasoning and reasoning != content:
            message["reasoning_content"] = reasoning
        if calls:
            message["tool_calls"] = calls
        messages.append(message)
        results = (step.get("observation") or {}).get("results") or []
        if calls and not results and index != len(steps) - 1:
            raise RejectedEpisode("missing tool observation")
        for result in results:
            messages.append({"role": "tool", "content": result["content"]})

    if not issue or len(positions) < 2:
        raise RejectedEpisode("missing issue or fewer than two decisions")

    task = record["task"]
    repository = task.rsplit("-", 1)[0].replace("__", "/", 1)
    model = record["model"]
    return Episode(
        task_id=f"swebench-verified:{task}",
        trajectory_id=f"{record['run_id']}:{trajectory['session_id']}",
        family="repository-repair",
        repository=repository,
        run_id=record["run_id"],
        run_timestamp=record["created_at"],
        source=f"{record['results_path']}#{record['artifact_uri']}",
        revision=BENCHMARK_REVISION,
        policy=model["name"],
        checkpoint=model["location"],
        checkpoint_revision=(model.get("config") or {}).get("revision"),
        policy_config_digest=model["config_digest"],
        goal=f"{REPAIR_CRITERION}\nRepository: {repository}\nIssue:\n{issue}",
        messages=messages,
        predict_before=positions,
        outcome=record["outcome"],
    )


def write_shard(path: Path, episodes: Iterable[Episode]) -> dict:
    """Publish one immutable trajectory shard and return split and label counts."""
    census: dict[str, Counter] = defaultdict(Counter)
    tasks = defaultdict(set)
    seen = set()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as stage:
        staged = Path(stage) / path.name
        with staged.open("w") as handle:
            for item in episodes:
                if item.trajectory_id in seen:
                    raise ValueError(f"duplicate trajectory {item.trajectory_id}")
                seen.add(item.trajectory_id)
                handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
                for key in (item.split, f"{item.split}/{item.policy}"):
                    counts = census[key]
                    counts["traces"] += 1
                    counts["turns"] += len(item.predict_before)
                    counts[f"outcome_{item.outcome}"] += 1
                    tasks[key].add(item.task_id)
        path.hardlink_to(staged)
    return {key: dict(counts, tasks=len(tasks[key])) for key, counts in census.items()}


def read_shards(paths: Iterable[Path]) -> Iterator[Episode]:
    """Read immutable shards, including shards added since the last run."""
    seen = set()
    for path in paths:
        with path.open() as handle:
            for line in handle:
                episode = Episode(**json.loads(line))
                if episode.trajectory_id in seen:
                    raise ValueError(f"duplicate trajectory {episode.trajectory_id}")
                seen.add(episode.trajectory_id)
                yield episode


def convert(sources: list[Path], output: Path) -> dict:
    rejected: dict[str, Counter] = defaultdict(Counter)

    def episodes() -> Iterator[Episode]:
        for source in sources:
            with source.open() as handle:
                for line in handle:
                    record = json.loads(line)
                    try:
                        yield harbor_episode(record)
                    except RejectedEpisode as error:
                        rejected[record["run_id"]][str(error)] += 1

    splits = write_shard(output, episodes())
    return {"splits": splits, "rejected": {run: dict(counts) for run, counts in rejected.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("source", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
