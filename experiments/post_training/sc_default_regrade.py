# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover missing judge verdicts from saved answers, without model generation."""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import pyarrow.parquet as pq
import yaml
from rigging.filesystem.buckets import filesystem_for
from rigging.secrets import require_secret
from taskcompendium.chat import chat_conversation
from taskcompendium.grading_result import Outcome
from taskcompendium.models import ConversationTrace, GradingAttempt, TaskSpec, TextMessage, grades_with_judge
from taskcompendium.pipeline.fingerprints import deduplication_key
from taskcompendium.runtime.task_grading import grade_task
from transformers import AutoTokenizer
from verifyit.modes.grade_judge import JudgeConnection

from experiments.post_training.sc_default_data import write_json


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(uri: str):
    filesystem, key = filesystem_for(uri)
    with filesystem.open(key, "rt") as stream:
        return json.load(stream)


def save_local(path: Path, value) -> None:
    """Keep the first valid verdict before any remote write is attempted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as destination:
        json.dump(value, destination, indent=2, default=lambda scalar: scalar.item())
        destination.flush()
        os.fsync(destination.fileno())
        temporary = Path(destination.name)
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def regrade(
    export: str,
    task_path: Path,
    step: int,
    adopted: Path | None = None,
    uids: set[str] | None = None,
    local_manifest: Path | None = None,
    publish_staged: bool = False,
) -> dict:
    fs, prefix = filesystem_for(export + f"/dumped_evals/global_step_{step}_evals")
    task_specs = pq.read_table(task_path, columns=["task_spec"]).column(0).to_pylist()
    manifest_uri = export + f"/study/regrades-{step:04d}.json"
    recovery_fs, recovery_key = filesystem_for(manifest_uri)
    recovery = read_json(manifest_uri) if recovery_fs.exists(recovery_key) else {"step": step, "rows": {}}
    if local_manifest is not None:
        if local_manifest.exists():
            staged = json.loads(local_manifest.read_text())
            if staged["export"] != export or staged["step"] != step:
                raise ValueError("Staged verdicts belong to a different run or step")
            for uid, value in recovery["rows"].items():
                if uid in staged["rows"] and staged["rows"][uid] != value:
                    raise ValueError("Staged verdict would replace an existing recovered verdict")
            staged["rows"].update(recovery["rows"])
            recovery = staged
        else:
            if publish_staged:
                raise ValueError("No local verdict manifest to publish")
            recovery["export"] = export
    if publish_staged:
        if local_manifest is None or not recovery["rows"]:
            raise ValueError("Publishing requires a nonempty local verdict manifest")
        if uids is not None and uids != set(recovery["rows"]):
            raise ValueError("Publication must validate every staged saved answer")
        uids = set(recovery["rows"])
    source_fs, source_key = filesystem_for(export.removesuffix("/exports") + "/resolved-launch.yaml")
    launch = yaml.safe_load(source_fs.cat_file(source_key))["config"]
    judge_model = launch["runtime"]["task_env"]["VERIFYIT_JUDGE_MODEL"]
    os.environ["VERIFYIT_JUDGE_MODEL"] = judge_model
    remote = launch["skyrl"]["environment"]["task_sessions"]["remote_judge"]
    connection = JudgeConnection(remote["base_url"], require_secret(remote["api_key_env"]))
    tokenizer = None
    adopted_grade = json.loads(adopted.read_text()) if adopted is not None else None
    seen = set()
    for key in sorted(fs.find(prefix)):
        if not key.endswith(".jsonl") or key.endswith("/aggregated_results.jsonl"):
            continue
        with fs.open(key, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                uid = str(row["uid"])
                if uids is not None and uid not in uids:
                    continue
                seen.add(uid)
                if (row.get("verification") or {}).get("status") == "verified":
                    if uid in recovery["rows"]:
                        raise ValueError("Recovery cannot replace an original valid grade")
                    continue
                if uid in recovery["rows"]:
                    previous = recovery["rows"][uid]
                    if previous["row_sha256"] != fingerprint(row):
                        raise ValueError("Previously recovered answer changed")
                    task = TaskSpec.model_validate_json(task_specs[int(uid)])
                    if previous["contract"] != deduplication_key(task):
                        raise ValueError("Previously recovered grading contract changed")
                    if previous["judge_model"] != judge_model or previous["judge_base_url"] != remote["base_url"]:
                        raise ValueError("Previously recovered judge configuration changed")
                    continue
                raw = task_specs[int(uid)]
                task = TaskSpec.model_validate_json(raw)
                if not grades_with_judge(task.grader):
                    raise ValueError(f"Non-judge failure at uid {uid} requires its original machine recovery")
                messages = row.get("grading_messages")
                reconstruction = "saved original grading messages"
                if messages is not None:
                    attempt = GradingAttempt(chat_conversation(messages), stop_reason=row["stop_reason"])
                else:
                    # Compatibility with the failed initial run before exact
                    # failed-attempt messages were added to the research dump.
                    if row["env_class"] != "taskcompendium" or task.context.events[-1].role == "assistant":
                        raise ValueError("This historical answer needs its original conversation evidence")
                    if tokenizer is None:
                        model = launch["inputs"]["model"]
                        tokenizer = AutoTokenizer.from_pretrained(
                            model["tokenizer_uri"], revision=model["tokenizer_revision"], local_files_only=True
                        )
                    answer = tokenizer.decode(row["response_token_ids"], skip_special_tokens=True)
                    reconstruction = "saved token IDs decoded with the serving API skip_special_tokens invariant"
                    attempt = GradingAttempt(
                        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=answer))),
                        stop_reason=row["stop_reason"],
                    )
                attempts = []
                if adopted_grade is not None and adopted_grade["uid"] == uid:
                    if adopted_grade["task_spec_sha256"] != hashlib.sha256(raw.encode()).hexdigest():
                        raise ValueError("Adopted recovery uses a different task contract")
                    if (
                        adopted_grade["response_token_ids_sha256"]
                        != hashlib.sha256(json.dumps(row["response_token_ids"]).encode()).hexdigest()
                    ):
                        raise ValueError("Adopted recovery uses a different saved answer")
                    if (
                        messages is None
                        and adopted_grade["answer_sha256"] != hashlib.sha256(answer.encode()).hexdigest()
                    ):
                        raise ValueError("Adopted recovery reconstructed different answer bytes")
                    verdict = adopted_grade["regrade"]
                    if verdict["status"] != Outcome.GRADED.value:
                        raise ValueError("Only the first valid already-recorded verdict may be adopted")
                    attempts.append({"adopted_proof_sha256": hashlib.sha256(adopted.read_bytes()).hexdigest()})
                else:
                    for _ in range(3):
                        started = time.perf_counter()
                        grade = grade_task(task, attempt, judge_connection=connection)
                        verdict = asdict(grade)
                        attempts.append({"seconds": time.perf_counter() - started, "verdict": verdict})
                        if grade.status is not Outcome.INFRA_ERROR:
                            break
                if verdict["status"] != Outcome.GRADED or verdict["reward"] is None:
                    pending = {"uid": uid, "attempts": attempts}
                    if local_manifest is not None:
                        save_local(local_manifest.with_name(local_manifest.name + ".pending"), pending)
                    else:
                        write_json(manifest_uri + ".pending", pending)
                    raise RuntimeError(f"Judge recovery still lacks a verdict at uid {uid}")
                recovery["rows"][uid] = {
                    "task": task.id,
                    "contract": deduplication_key(task),
                    "row_sha256": fingerprint(row),
                    "recovery_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "judge_model": judge_model,
                    "judge_base_url": remote["base_url"],
                    "original_verification": row.get("verification"),
                    "attempts": attempts,
                    "answer_reconstruction": reconstruction,
                    "verification": {
                        "status": "verified",
                        "score": verdict["reward"],
                        "score_min": 0.0,
                        "score_max": 1.0,
                        "passed": None,
                        "reason": None,
                        "diagnostics": {"grade_only_recovery": True},
                    },
                }
                if local_manifest is not None:
                    save_local(local_manifest, recovery)
                else:
                    write_json(manifest_uri, recovery)
    if uids is not None and uids - seen:
        raise ValueError(f"Requested saved answer UIDs were not found: {sorted(uids - seen)}")
    if publish_staged:
        write_json(manifest_uri, recovery)
    return recovery


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export")
    parser.add_argument("tasks", type=Path)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument(
        "--uids", nargs="+", help="Retry only these saved answer UIDs; complete analysis coverage remains required"
    )
    parser.add_argument(
        "--adopt", type=Path, help="Keep the first valid verdict already recorded during operator recovery"
    )
    parser.add_argument(
        "--local-manifest", type=Path, help="Durably stage verdicts locally without writing remote storage"
    )
    parser.add_argument(
        "--publish-staged", action="store_true", help="Validate saved-answer hashes and publish the local verdicts"
    )
    args = parser.parse_args()
    selected = set(args.uids) if args.uids is not None else None
    print(
        json.dumps(
            regrade(
                args.export.rstrip("/"),
                args.tasks,
                args.step,
                args.adopt,
                selected,
                args.local_manifest,
                args.publish_staged,
            ),
            indent=2,
        )
    )
