# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit frozen grading contracts without using model answers or changing tasks."""

import argparse
import base64
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq


def resource(task: dict, name: str) -> bytes:
    matches = [row for group in task["resources"].values() for row in group if row["path"] == name]
    if len(matches) != 1 or matches[0]["source"]["kind"] != "inline_file":
        raise ValueError(f"Expected one original inline resource: {name}")
    return base64.b64decode(matches[0]["source"]["content_base64"])


def references(value):
    if isinstance(value, dict):
        return [part for key, part in value.items() if key == "$ref"] + [
            item for part in value.values() for item in references(part)
        ]
    if isinstance(value, list):
        return [item for part in value for item in references(part)]
    return []


def differences(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        return [
            item
            for key in sorted(left.keys() | right.keys())
            for item in differences(left.get(key), right.get(key), path + "/" + str(key))
        ]
    return [] if left == right else [path]


def audit(tasks_path: Path) -> dict:
    tasks = [json.loads(raw) for raw in pq.read_table(tasks_path, columns=["task_spec"]).column(0).to_pylist()]
    schemas, intervals = [], []
    for uid, task in enumerate(tasks):
        if task["grader"].get("mode") == "json-schema":
            raw = resource(task, task["grader"]["parameters"]["schema"])
            schema = json.loads(raw)
            unresolved = []
            for ref in references(schema):
                if not ref.startswith("#/"):
                    continue
                value = schema
                try:
                    for token in ref[2:].split("/"):
                        token = token.replace("~1", "/").replace("~0", "~")
                        value = value[int(token)] if isinstance(value, list) else value[token]
                except (KeyError, IndexError, TypeError, ValueError):
                    unresolved.append(ref)
            if unresolved:
                schemas.append(
                    {
                        "uid": str(uid),
                        "task": task["id"],
                        "schema_sha256": hashlib.sha256(raw).hexdigest(),
                        "unresolved_local_refs": sorted(set(unresolved)),
                    }
                )
        if task["id"].startswith("reasoning_gym_generated-"):
            contract = json.loads(resource(task, "config.json"))["contract"]
            if contract["task"] == "time_intervals":
                intervals.append((str(uid), task, contract))

    if not intervals or "generate" in sys.modules:
        raise ValueError("Expected time-interval contracts and a fresh audit process")
    version = importlib.metadata.version("reasoning-gym")
    if version != "0.1.25":
        raise ValueError(f"The original grader requires reasoning-gym 0.1.25, got {version}")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("Run the original generation replay with PYTHONHASHSEED=0")
    script = resource(intervals[0][1], "grade.py")
    generator = resource(intervals[0][1], "generate.py")
    replay = []
    with TemporaryDirectory(prefix="sc-grader-audit-") as directory:
        folder = Path(directory)
        for name in ("grade.py", "generate.py"):
            (folder / name).write_bytes(resource(intervals[0][1], name))
        previous_path = sys.path[:]
        sys.path.insert(0, directory)
        try:
            spec = importlib.util.spec_from_file_location("sc_frozen_grader", folder / "grade.py")
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            with contextlib.redirect_stdout(io.StringIO()):
                spec.loader.exec_module(module)
            for uid, task, contract in intervals:
                if resource(task, "grade.py") != script or resource(task, "generate.py") != generator:
                    raise ValueError("Time-interval contracts use different original scripts")
                with contextlib.redirect_stdout(io.StringIO()):
                    actual = module.encoded(module.regenerated_entry(contract["generation"]))
                expected = contract["entry"]
                replay.append(
                    {
                        "uid": uid,
                        "task": task["id"],
                        "task_type": expected["metadata"]["task_type"],
                        "configuration_check_passed": True,
                        "differing_fields": differences(expected, actual),
                        "question_matches": actual["question"] == expected["question"],
                        "answer_matches": actual["answer"] == expected["answer"],
                        "recorded_date": expected["metadata"]["start_time"][:10],
                        "current_date": actual["metadata"]["start_time"][:10],
                    }
                )
        finally:
            sys.path[:] = previous_path
    return {
        "validation_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "all_tasks": len(tasks),
        "unresolved_schemas": schemas,
        "reasoning_gym_version": version,
        "python_hash_seed": 0,
        "original_grader_script_sha256": hashlib.sha256(script).hexdigest(),
        "time_interval_replays": replay,
        "model_answers_used": False,
        "scoring_called": False,
        "exclusion_applied": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.tasks)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "unresolved_schemas": len(result["unresolved_schemas"]),
                "time_interval_replays": len(result["time_interval_replays"]),
            }
        )
    )
