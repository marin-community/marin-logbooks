# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the training arms of the Open-SWE outcome-filter ablation.

Three steps run over the raw Open-SWE-Traces downloads that Marin's adapter already defines:

1. ``index`` records one small row per trajectory: partition, outcome, size and whether
   Marin's adapter keeps it.
2. ``membership`` selects the trajectory IDs of every arm from that index.
3. ``arm`` writes one arm as OpenAI-style chat rows that the SFT launcher tokenizes.

Arms hold only trajectories the adapter keeps and that fit the training window when rendered
with the training chat template, so no arm is truncated. ``solved-*`` arms hold solved
trajectories. ``random-*`` arms draw from every outcome, with the same partition mix and the
same token count per partition as the solved arm of that size, so the pair differs in outcome
alone.

Tracked in marin-community/marin-logbooks#39.
"""

import hashlib
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from levanter.data.text.formats import ChatProcessor
from levanter.tokenizers import load_tokenizer
from marin.datakit.download.open_swe_traces import (
    OPEN_SWE_TRACES_PARTITIONS,
    QWEN38_SOURCES,
    excluded_trajectory_ids,
    link_tool_call_ids,
    open_swe_traces_chat_normalize_steps,
    restore_solution_originality,
    row_to_chat_doc,
)
from marin.datakit.download.rollout_transforms import load_parquet_batched
from marin.execution.step_spec import StepSpec
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.sft.opencode_chat_template import OPENCODE_TOOLS_CHAT_TEMPLATE

logger = logging.getLogger(__name__)

STEP_PREFIX = "open-swe-outcome"
INDEX_VERSION = "2026.11.13.2"
MEMBERSHIP_VERSION = "2026.11.13.2"
ARM_VERSION = "2026.11.13.2"
SOLVED = 1
TOKENIZER = "Qwen/Qwen3-8B"
# The training window. Levanter's GPU attention fallback runs out of memory at 131,072 tokens.
MAX_TOKENS = 65_536
RUNG_ROWS = (10_000, 31_600)
SOLVED_SALT = "open-swe-outcome/solved"
RANDOM_SALT = "open-swe-outcome/random"
STATUS_KEPT = "kept"
STATUS_EXCLUDED = "excluded"
STATUS_DROPPED = "dropped"
WORKER_RESOURCES = ResourceConfig(cpu=1, ram="32g")

ARM_SOLVED_ALL = "solved-all"
ARM_RANDOM_ALL = "random-all"
ARM_ALL = "all"


def solved_arm(rows: int) -> str:
    return f"solved-{rows}"


def random_arm(rows: int) -> str:
    return f"random-{rows}"


ARMS = (
    *(name(rows) for rows in RUNG_ROWS for name in (solved_arm, random_arm)),
    ARM_SOLVED_ALL,
    ARM_RANDOM_ALL,
    ARM_ALL,
)

INDEX_SCHEMA = pa.schema(
    [
        pa.field("trajectory_id", pa.string()),
        pa.field("partition", pa.string()),
        pa.field("resolved", pa.int32()),
        pa.field("num_messages", pa.int32()),
        pa.field("num_chars", pa.int64()),
        pa.field("num_tokens", pa.int64()),
        pa.field("status", pa.string()),
    ]
)


@dataclass(frozen=True)
class IndexRow:
    trajectory_id: str
    partition: str
    resolved: int
    num_tokens: int


@dataclass(frozen=True)
class RawPartition:
    name: str
    raw: StepSpec
    exclusions: StepSpec | None


def raw_partitions() -> list[RawPartition]:
    """The adapter's own raw-download steps, so the arms read the pinned source revision."""
    partitions = []
    for name in OPEN_SWE_TRACES_PARTITIONS:
        deps = open_swe_traces_chat_normalize_steps(name)[0].deps
        partitions.append(RawPartition(name=name, raw=deps[0], exclusions=deps[1] if len(deps) > 1 else None))
    return partitions


def message_chars(message: dict) -> int:
    """Characters of text the chat template renders for one message."""
    total = len(message.get("content") or "") + len(message.get("reasoning_content") or "")
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        total += len(function.get("name") or "") + len(function.get("arguments") or "")
    return total


def repaired_row(row: dict, partition: str) -> dict:
    return restore_solution_originality(row) if partition in QWEN38_SOURCES else row


@cache
def training_chat_processor() -> ChatProcessor:
    """The processor the SFT launcher tokenizes with, so index token counts match training."""
    return ChatProcessor(load_tokenizer(TOKENIZER), chat_template=OPENCODE_TOOLS_CHAT_TEMPLATE)


def index_rows(path: str, partition: str, excluded_ids: frozenset[str]) -> Iterator[dict]:
    """One index row per trajectory of a raw file, with the adapter's keep/drop verdict."""
    for row in load_parquet_batched(path):
        messages = row.get("messages") or []
        num_tokens = 0
        if row["trajectory_id"] in excluded_ids:
            status = STATUS_EXCLUDED
        elif row_to_chat_doc(repaired_row(row, partition)):
            status = STATUS_KEPT
            num_tokens = len(training_chat_processor()([chat_row(row, partition)])[0]["input_ids"])
        else:
            status = STATUS_DROPPED
        yield {
            "trajectory_id": row["trajectory_id"],
            "partition": partition,
            "resolved": row["resolved"],
            "num_messages": len(messages),
            "num_chars": sum(message_chars(message) for message in messages),
            "num_tokens": num_tokens,
            "status": status,
        }


def chat_row(row: dict, partition: str) -> dict:
    """An OpenAI-style chat row with the tool definitions as chat-template arguments."""
    repaired = repaired_row(row, partition)
    # Null fields would render as the string "None" in a chat template.
    messages = [{key: value for key, value in message.items() if value is not None} for message in repaired["messages"]]
    return {
        "messages": link_tool_call_ids(messages),
        "chat_template_kwargs": {"tools": [json.loads(tool) for tool in repaired["tools"]]},
        "trajectory_id": row["trajectory_id"],
        "partition": partition,
        "resolved": row["resolved"],
    }


def hash_order(rows: list[IndexRow], salt: str) -> list[IndexRow]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{salt}:{row.trajectory_id}".encode()).hexdigest())


def volume_matched(pool: list[IndexRow], target_tokens: int, salt: str) -> list[IndexRow]:
    """The shortest hash-ordered prefix of ``pool`` holding at least ``target_tokens``."""
    selected: list[IndexRow] = []
    tokens = 0
    for row in hash_order(pool, salt):
        if tokens >= target_tokens:
            break
        selected.append(row)
        tokens += row.num_tokens
    return selected


def partition_matched_random(solved: list[IndexRow], eligible: list[IndexRow]) -> list[IndexRow]:
    """Rows of any outcome matching ``solved`` in partition mix and per-partition token count."""
    target: Counter[str] = Counter()
    for row in solved:
        target[row.partition] += row.num_tokens
    pools: defaultdict[str, list[IndexRow]] = defaultdict(list)
    for row in eligible:
        pools[row.partition].append(row)
    selected: list[IndexRow] = []
    for partition in sorted(target):
        selected.extend(volume_matched(pools[partition], target[partition], RANDOM_SALT))
    return selected


def arm_members(kept: list[IndexRow], rung_rows: tuple[int, ...], max_tokens: int) -> dict[str, list[IndexRow]]:
    """Select every arm's trajectories from the kept rows that fit ``max_tokens``."""
    eligible = [row for row in kept if row.num_tokens <= max_tokens]
    solved_all = hash_order([row for row in eligible if row.resolved == SOLVED], SOLVED_SALT)
    if len(solved_all) < max(rung_rows):
        raise ValueError(f"only {len(solved_all)} solved trajectories fit; the largest rung needs {max(rung_rows)}")
    members: dict[str, list[IndexRow]] = {}
    for rows in rung_rows:
        members[solved_arm(rows)] = solved_all[:rows]
        members[random_arm(rows)] = partition_matched_random(solved_all[:rows], eligible)
    members[ARM_SOLVED_ALL] = solved_all
    members[ARM_RANDOM_ALL] = partition_matched_random(solved_all, eligible)
    members[ARM_ALL] = eligible
    return members


def arm_summary(rows: list[IndexRow]) -> dict:
    by_outcome: Counter[str] = Counter(str(row.resolved) for row in rows)
    by_partition: Counter[str] = Counter(row.partition for row in rows)
    tokens_by_partition: Counter[str] = Counter()
    for row in rows:
        tokens_by_partition[row.partition] += row.num_tokens
    return {
        "rows": len(rows),
        "tokens": sum(row.num_tokens for row in rows),
        "rows_by_resolved": dict(sorted(by_outcome.items())),
        "rows_by_partition": dict(sorted(by_partition.items())),
        "tokens_by_partition": dict(sorted(tokens_by_partition.items())),
    }


def raw_files(partitions: list[RawPartition]) -> list[tuple[str, str]]:
    files = [
        (str(path), partition.name)
        for partition in partitions
        for path in (StoragePath.parse(partition.raw.output_path) / "**/*.parquet").glob()
    ]
    if not files:
        raise ValueError("no raw Open-SWE-Traces parquet files found")
    return files


def all_excluded_ids(partitions: list[RawPartition]) -> frozenset[str]:
    exclusion_paths = {partition.exclusions.output_path for partition in partitions if partition.exclusions is not None}
    return frozenset().union(*(excluded_trajectory_ids(path) for path in exclusion_paths))


def read_index(index_path: str) -> pa.Table:
    tables = []
    for path in (StoragePath.parse(index_path) / "*.parquet").glob():
        with path.open("rb") as source:
            tables.append(pq.read_table(source))
    return pa.concat_tables(tables)


def index_step(partitions: list[RawPartition]) -> StepSpec:
    deps = [partition.raw for partition in partitions]
    deps += list({id(p.exclusions): p.exclusions for p in partitions if p.exclusions is not None}.values())

    def build(output_path: str) -> None:
        excluded_ids = all_excluded_ids(partitions)
        pipeline = (
            Dataset.from_list(raw_files(partitions))
            .flat_map(lambda item: index_rows(item[0], item[1], excluded_ids))
            .write_parquet(prefix_join(output_path, "index-{shard:05d}-of-{total:05d}.parquet"), schema=INDEX_SCHEMA)
        )
        ZephyrContext(name="open-swe-outcome-index", resources=WORKER_RESOURCES).execute(pipeline)

    return StepSpec(name=f"{STEP_PREFIX}/index", deps=deps, fn=build, hash_attrs={"version": INDEX_VERSION})


def membership_step(index: StepSpec) -> StepSpec:
    def build(output_path: str) -> None:
        table = read_index(index.output_path)
        statuses = Counter(zip(table.column("status").to_pylist(), table.column("resolved").to_pylist(), strict=True))
        kept = [
            IndexRow(row["trajectory_id"], row["partition"], row["resolved"], row["num_tokens"])
            for row in table.filter(pc.equal(table.column("status"), STATUS_KEPT)).to_pylist()
        ]
        if len({row.trajectory_id for row in kept}) != len(kept):
            raise ValueError("trajectory IDs are not unique among kept rows")
        members = arm_members(kept, RUNG_ROWS, MAX_TOKENS)
        summary = {
            "index_rows": table.num_rows,
            "max_tokens": MAX_TOKENS,
            "kept_rows_within_max_tokens_by_resolved": dict(
                sorted(Counter(str(row.resolved) for row in kept if row.num_tokens <= MAX_TOKENS).items())
            ),
            "rows_by_status_and_resolved": {
                f"{status}/{resolved}": count for (status, resolved), count in statuses.items()
            },
            "arms": {arm: arm_summary(rows) for arm, rows in members.items()},
        }
        root = StoragePath.parse(output_path)
        for arm, rows in members.items():
            with (root / f"{arm}.parquet").open("wb") as sink:
                pq.write_table(pa.table({"trajectory_id": [row.trajectory_id for row in rows]}), sink)
        (root / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
        logger.info("arm sizes: %s", {arm: len(rows) for arm, rows in members.items()})

    return StepSpec(
        name=f"{STEP_PREFIX}/membership",
        deps=[index],
        fn=build,
        hash_attrs={"version": MEMBERSHIP_VERSION, "rung_rows": list(RUNG_ROWS), "max_tokens": MAX_TOKENS},
    )


def arm_rows(path: str, partition: str, member_ids: frozenset[str]) -> Iterator[dict]:
    for row in load_parquet_batched(path):
        if row["trajectory_id"] in member_ids:
            yield chat_row(row, partition)


def arm_step(arm: str, membership: StepSpec, partitions: list[RawPartition]) -> StepSpec:
    def build(output_path: str) -> None:
        with (StoragePath.parse(membership.output_path) / f"{arm}.parquet").open("rb") as source:
            member_ids = frozenset(pq.read_table(source).column("trajectory_id").to_pylist())
        pipeline = (
            Dataset.from_list(raw_files(partitions))
            .flat_map(lambda item: arm_rows(item[0], item[1], member_ids))
            .write_jsonl(prefix_join(output_path, "train-{shard:05d}-of-{total:05d}.jsonl.gz"))
        )
        ZephyrContext(name=f"open-swe-outcome-arm-{arm}", resources=WORKER_RESOURCES).execute(pipeline)

    return StepSpec(
        name=f"{STEP_PREFIX}/arms/{arm}",
        deps=[membership, *(partition.raw for partition in partitions)],
        fn=build,
        hash_attrs={"version": ARM_VERSION, "arm": arm, "max_tokens": MAX_TOKENS},
    )
