# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze a new task holdout from the published five-family Snowball pool."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.buckets import filesystem_for
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.fingerprints import deduplication_key, semantic_digest

SOURCE_URI = "s3://marin-us-east-02a/marin/rl-data/task-selection/snowball-100k/2026.10.23.2"
SOURCE_SHA256 = "bc9326bc87e9fd992748147aaa3f8261e649819456027645cb00cbc23dc7e94d"
SPLIT_SEED = 20261028
FAMILIES = ("calendar", "competitive_coding", "reasoning_gym", "stem", "structural")
SOURCE_FAMILIES = {
    "eurus2_code": "competitive_coding",
    "nemotron_ultra_rlvr1_ultra_sft_step3200_calendar_v2": "calendar",
    "nemotron_ultra_rlvr1_ultra_sft_step3200_comp_coding": "competitive_coding",
    "nemotron_ultra_rlvr1_ultra_sft_step3200_reasoning_gym": "reasoning_gym",
    "nemotron_ultra_rlvr1_ultra_sft_step3200_stem_mcqa": "stem",
    "nemotron_ultra_rlvr1_ultra_sft_step3200_structured_outputs_v2": "structural",
    "openscience": "stem",
    "reasoning_gym_generated": "reasoning_gym",
    "snowball-v01-structure-closure-train": "structural",
    "tasktrove-glaive_code": "competitive_coding",
    "tasktrove-calendar": "calendar",
    "tasktrove-if_calendar": "calendar",
    "tasktrove-structured": "structural",
}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_json(uri: str):
    filesystem, path = filesystem_for(uri)
    with filesystem.open(path, "rt") as source:
        return json.load(source)


def write_json(uri: str, value) -> None:
    filesystem, path = filesystem_for(uri)
    with filesystem.open(path, "wt") as destination:
        json.dump(value, destination, indent=2, default=lambda scalar: scalar.item())


def prepare_study_data(source_path: str, output_path: str, smoke: bool) -> None:
    """Preserve canonical tasks, excluding every selected public/contract duplicate from training."""
    filesystem, source_path = filesystem_for(source_path.rstrip("/") + "/train.parquet")
    groups = defaultdict(list)
    contracts, source_counts = set(), Counter()
    source_digest = hashlib.sha256()
    with TemporaryDirectory() as temporary:
        local = Path(temporary) / "source.parquet"
        with filesystem.open(source_path, "rb") as source, local.open("wb") as destination:
            while chunk := source.read(8 * 1024 * 1024):
                source_digest.update(chunk)
                destination.write(chunk)
        if source_digest.hexdigest() != SOURCE_SHA256:
            raise ValueError("Published Snowball source bytes changed")
        for batch in pq.ParquetFile(local).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                task = TaskSpec.model_validate_json(row["task_spec"])
                source = task.id.rsplit("-", 1)[0]
                family = SOURCE_FAMILIES[source]
                public = semantic_digest(task, False)
                contract = deduplication_key(task)
                if contract in contracts:
                    raise ValueError(f"Duplicate source contract: {task.id}")
                contracts.add(contract)
                groups[family].append(
                    {
                        "task_spec": row["task_spec"],
                        "teacher_route": family,
                        "study_id": task.id,
                        "source": source,
                        "public": public,
                        "contract": contract,
                    }
                )
                source_counts[source] += 1
    if sum(source_counts.values()) != 100_000:
        raise ValueError("Published Snowball row count changed")

    selected, validation = set(), {}
    for family in FAMILIES:
        rows = sorted(groups[family], key=lambda row: digest(f"{SPLIT_SEED}:{row['study_id']}"))
        unique = {}
        for row in rows:
            unique.setdefault(row["public"], row)
        panel = list(unique.values())[:2000]
        if len(panel) != 2000:
            raise ValueError(f"Insufficient distinct public tasks in {family}")
        validation[family] = panel
        selected.update(row["public"] for row in panel)
    training = [row for family in FAMILIES for row in groups[family] if row["public"] not in selected]
    # Interleave families so the fixed 1,000-task curve panel is a balanced prefix.
    evaluation = [
        row for positions in zip(*(validation[family] for family in FAMILIES), strict=True) for row in positions
    ]
    if smoke:
        # Exercise every source's grader and both train/eval paths with real tasks.
        by_source = defaultdict(list)
        for row in training:
            by_source[row["source"]].append(row)
        training = [row for source in sorted(by_source) for row in by_source[source][:8]]
        evaluation = evaluation[:10]

    manifest = {
        "source_uri": SOURCE_URI,
        "source_sha256": SOURCE_SHA256,
        "source_rows": sum(source_counts.values()),
        "source_counts": dict(source_counts),
        "split_seed": SPLIT_SEED,
        "selection": (
            "SHA256(seed:task_id), first 2,000 distinct public fingerprints per family; "
            "all matching public fingerprints excluded from training"
        ),
        "smoke": smoke,
        "rows": {"train": len(training), "validation": len(evaluation)},
        "family_counts": {
            label: dict(Counter(row["teacher_route"] for row in rows))
            for label, rows in (("train", training), ("validation", evaluation))
        },
        "public_overlap": len({row["public"] for row in training} & {row["public"] for row in evaluation}),
        "contract_overlap": len({row["contract"] for row in training} & {row["contract"] for row in evaluation}),
        "limits": [
            "A fresh RL holdout from the frozen representative curriculum; earlier SFT/pretraining exposure is unknown.",
            "Exact public and full-contract separation does not establish semantic decontamination.",
            "The earlier 910-task model-selection panel is not used for the primary endpoint.",
        ],
    }
    for label, rows in (("train", training), ("validation", evaluation)):
        table = pa.Table.from_pylist([{"task_spec": row["task_spec"]} for row in rows])
        with TemporaryDirectory() as temporary:
            local = Path(temporary) / f"{label}.parquet"
            pq.write_table(table, local)
            with local.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            destination_fs, destination = filesystem_for(output_path.rstrip("/") + f"/{label}.parquet")
            destination_fs.put_file(str(local), destination)
        manifest[f"{label}_sha256"] = checksum
        manifest[f"{label}_ids_sha256"] = digest("\n".join(row["study_id"] for row in rows))
    write_json(output_path.rstrip("/") + "/selection.json", manifest)
    write_json(
        output_path.rstrip("/") + "/evaluation-index.json",
        [
            {
                "uid": str(index),
                **{key: row[key] for key in ("study_id", "teacher_route", "source", "public", "contract")},
            }
            for index, row in enumerate(evaluation)
        ],
    )
