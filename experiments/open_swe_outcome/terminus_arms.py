# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Open-SWE outcome-ablation arms in Harbor's Terminus-2 format.

The arms follow ``arms.py``'s design. The solved arm holds the leading solved trajectories in
``SOLVED_SALT`` hash order. The random arm draws every outcome in ``RANDOM_SALT`` hash order,
matching the solved arm's partition mix and per-partition token volume. Here a trajectory is
eligible when ``terminus.terminus_row`` converts it and its Terminus-2 rendering fits the
65,536-token training window. Token counts use that rendering, so the arms are matched in the
tokens actually trained.

The candidates are the 31,600-row arms that ``arms.py`` materialized at version 2026.11.13.1,
with a 131,072-token window. Each is the leading part of its hash order, so the first eligible
rows of a pool are the rows a rebuild from every Open-SWE trajectory would select. Membership
fails if a partition's pool runs out before its target volume.

Tracked in marin-community/marin-logbooks#39.
"""

import json
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from functools import cache

import pyarrow as pa
import pyarrow.parquet as pq
from levanter.data.text.formats import ChatProcessor
from levanter.tokenizers import load_tokenizer
from marin.execution.step_spec import StepSpec
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_jsonl

from experiments.open_swe_outcome.arms import (
    MAX_TOKENS,
    RANDOM_SALT,
    SOLVED,
    SOLVED_SALT,
    STEP_PREFIX,
    TOKENIZER,
    WORKER_RESOURCES,
    IndexRow,
    arm_summary,
    hash_order,
    volume_matched,
)
from experiments.open_swe_outcome.terminus import Unconvertible, terminus_row
from experiments.open_swe_outcome.terminus_chat_template import TERMINUS_CHAT_TEMPLATE

logger = logging.getLogger(__name__)

VERSION = "2026.11.17"
ARM_ROWS = 10_000
STATUS_KEPT = "kept"
STATUS_UNCONVERTIBLE = "unconvertible"


class Pool(StrEnum):
    SOLVED = "solved"
    RANDOM = "random"


POOL_ROOT = "s3://marin-us-east-08a/marin/open-swe-outcome/arms"
POOL_PATHS = {
    Pool.SOLVED: f"{POOL_ROOT}/solved-31600_b38fb7ab",
    Pool.RANDOM: f"{POOL_ROOT}/random-31600_95fc5b32",
}
ARM_POOLS = {f"terminus-solved-{ARM_ROWS}": Pool.SOLVED, f"terminus-random-{ARM_ROWS}": Pool.RANDOM}
TERMINUS_ARMS = tuple(ARM_POOLS)

INDEX_SCHEMA = pa.schema(
    [
        pa.field("trajectory_id", pa.string()),
        pa.field("partition", pa.string()),
        pa.field("resolved", pa.int32()),
        pa.field("pool", pa.string()),
        pa.field("num_tokens", pa.int64()),
        pa.field("status", pa.string()),
        pa.field("reason", pa.string()),
    ]
)


@dataclass(frozen=True)
class IndexedRow:
    row: IndexRow
    pool: Pool


@cache
def terminus_chat_processor() -> ChatProcessor:
    return ChatProcessor(load_tokenizer(TOKENIZER), chat_template=TERMINUS_CHAT_TEMPLATE)


def pool_files() -> list[tuple[str, Pool]]:
    files = [
        (str(path), pool)
        for pool, root in POOL_PATHS.items()
        for path in (StoragePath.parse(root) / "*.jsonl.gz").glob()
    ]
    if not files:
        raise ValueError("no pool files found")
    return files


def index_rows(path: str, pool: Pool) -> Iterator[dict]:
    for row in load_jsonl(path):
        record = {
            "trajectory_id": row["trajectory_id"],
            "partition": row["partition"],
            "resolved": row["resolved"],
            "pool": pool.value,
            "num_tokens": 0,
            "status": STATUS_KEPT,
            "reason": "",
        }
        try:
            converted = terminus_row(row)
        except Unconvertible as error:
            yield {**record, "status": STATUS_UNCONVERTIBLE, "reason": str(error)}
            continue
        yield {**record, "num_tokens": len(terminus_chat_processor()([converted])[0]["input_ids"])}


def matched_random(solved: list[IndexRow], eligible: list[IndexRow]) -> list[IndexRow]:
    """``arms.partition_matched_random``, failing when a partition's pool cannot reach its volume."""
    target: Counter[str] = Counter()
    for row in solved:
        target[row.partition] += row.num_tokens
    selected: list[IndexRow] = []
    for partition in sorted(target):
        pool = [row for row in eligible if row.partition == partition]
        chosen = volume_matched(pool, target[partition], RANDOM_SALT)
        if sum(row.num_tokens for row in chosen) < target[partition]:
            raise ValueError(f"random pool for {partition} holds too few tokens")
        selected.extend(chosen)
    return selected


def terminus_members(rows: list[IndexedRow]) -> dict[str, list[IndexRow]]:
    def eligible(pool: Pool) -> list[IndexRow]:
        return [entry.row for entry in rows if entry.pool is pool and entry.row.num_tokens <= MAX_TOKENS]

    solved_pool = [row for row in eligible(Pool.SOLVED) if row.resolved == SOLVED]
    solved = hash_order(solved_pool, SOLVED_SALT)[:ARM_ROWS]
    if len(solved) < ARM_ROWS:
        raise ValueError(f"only {len(solved_pool)} eligible solved trajectories; the arm needs {ARM_ROWS}")
    return {
        f"terminus-solved-{ARM_ROWS}": solved,
        f"terminus-random-{ARM_ROWS}": matched_random(solved, eligible(Pool.RANDOM)),
    }


def index_step() -> StepSpec:
    def build(output_path: str) -> None:
        pipeline = (
            Dataset.from_list(pool_files())
            .flat_map(lambda item: index_rows(item[0], item[1]))
            .write_parquet(prefix_join(output_path, "index-{shard:05d}-of-{total:05d}.parquet"), schema=INDEX_SCHEMA)
        )
        ZephyrContext(name="open-swe-terminus-index", resources=WORKER_RESOURCES).execute(pipeline)

    return StepSpec(
        name=f"{STEP_PREFIX}/terminus/index",
        fn=build,
        hash_attrs={"version": VERSION, "pools": {pool.value: path for pool, path in POOL_PATHS.items()}},
    )


def membership_step(index: StepSpec) -> StepSpec:
    def build(output_path: str) -> None:
        tables = []
        for path in (StoragePath.parse(index.output_path) / "*.parquet").glob():
            with path.open("rb") as source:
                tables.append(pq.read_table(source))
        records = pa.concat_tables(tables).to_pylist()
        kept = [
            IndexedRow(IndexRow(r["trajectory_id"], r["partition"], r["resolved"], r["num_tokens"]), Pool(r["pool"]))
            for r in records
            if r["status"] == STATUS_KEPT
        ]
        members = terminus_members(kept)
        summary = {
            "max_tokens": MAX_TOKENS,
            "rows_by_pool_and_status": dict(Counter(f"{r['pool']}/{r['status']}" for r in records)),
            "unconvertible_reasons": dict(Counter(r["reason"] for r in records if r["status"] != STATUS_KEPT)),
            "kept_over_max_tokens_by_pool_and_resolved": dict(
                Counter(f"{e.pool}/{e.row.resolved}" for e in kept if e.row.num_tokens > MAX_TOKENS)
            ),
            "arms": {arm: arm_summary(rows) for arm, rows in members.items()},
        }
        root = StoragePath.parse(output_path)
        for arm, rows in members.items():
            with (root / f"{arm}.parquet").open("wb") as sink:
                pq.write_table(pa.table({"trajectory_id": [row.trajectory_id for row in rows]}), sink)
        (root / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
        logger.info("terminus arm sizes: %s", {arm: len(rows) for arm, rows in members.items()})

    return StepSpec(
        name=f"{STEP_PREFIX}/terminus/membership",
        deps=[index],
        fn=build,
        hash_attrs={"version": VERSION, "arm_rows": ARM_ROWS, "max_tokens": MAX_TOKENS},
    )


def arm_rows(path: str, member_ids: frozenset[str]) -> Iterator[dict]:
    for row in load_jsonl(path):
        if row["trajectory_id"] in member_ids:
            yield terminus_row(row)


def arm_step(arm: str, membership: StepSpec) -> StepSpec:
    pool = ARM_POOLS[arm]

    def build(output_path: str) -> None:
        with (StoragePath.parse(membership.output_path) / f"{arm}.parquet").open("rb") as source:
            member_ids = frozenset(pq.read_table(source).column("trajectory_id").to_pylist())
        files = [path for path, file_pool in pool_files() if file_pool is pool]
        pipeline = (
            Dataset.from_list(files)
            .flat_map(lambda path: arm_rows(path, member_ids))
            .write_jsonl(prefix_join(output_path, "train-{shard:05d}-of-{total:05d}.jsonl.gz"))
        )
        ZephyrContext(name=f"open-swe-{arm}", resources=WORKER_RESOURCES).execute(pipeline)

    return StepSpec(
        name=f"{STEP_PREFIX}/terminus/arms/{arm}",
        deps=[membership],
        fn=build,
        hash_attrs={"version": VERSION, "arm": arm},
    )


def terminus_arm_steps() -> dict[str, StepSpec]:
    membership = membership_step(index_step())
    return {arm: arm_step(arm, membership) for arm in TERMINUS_ARMS}
