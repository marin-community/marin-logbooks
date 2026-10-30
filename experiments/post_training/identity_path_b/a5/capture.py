# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Four disjoint one-sample shards using the byte-pinned proven raw T2 primitive."""

import argparse
import hashlib
import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.identity_path_b.a5.data import SHARDS, SOURCES, TEACHER_SYSTEM
from experiments.post_training.identity_path_b.checkpoints import TEMPORARY_PREFIX
from experiments.post_training.identity_path_b.replay_capture import SOURCE_MODEL, T2_CAPTURE_SHA256
from experiments.post_training.identity_t2 import capture as t2


def checked_manifest(path: Path, expected_sha256: str, shard: int) -> list[dict]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("A5 manifest changed")
    if hashlib.sha256(Path(t2.__file__).read_bytes()).hexdigest() != T2_CAPTURE_SHA256:
        raise ValueError("Known-good T2 capture primitive changed")
    rows = [json.loads(line) for line in raw.splitlines()]
    if [row["global_index"] for row in rows] != list(range(shard, SOURCES, SHARDS)):
        raise ValueError("A5 shard must contain every original assigned index exactly once")
    if len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("Repeated original in A5 generation shard")
    for row in rows:
        if (
            row["teacher_system"] != TEACHER_SYSTEM
            or row["teacher"] != "held_profile_user_only"
            or row["tools"]
            or row["enable_thinking"] is not True
            or row["shard"] != shard
            or not row["prompt"]
            or any(message["role"] != "user" or set(message) != {"role", "content"} for message in row["prompt"])
        ):
            raise ValueError("A5 teacher/system/ordered USER-only context changed")
    return rows


def capture(rows: list[dict], *, endpoint: str, tokenizer, output: Path, artifact_root: str) -> dict:
    if not endpoint.startswith("https://") or not endpoint.endswith("/v1"):
        raise ValueError("Require an owned native TLS endpoint ending/v1")
    if not artifact_root.startswith(TEMPORARY_PREFIX):
        raise ValueError("A5 raw streams require RNO14d")
    remote = StoragePath(artifact_root) / "raw"
    if remote.exists():
        raise FileExistsError("A5 raw stream prefix already exists; no resume or retry")
    output.mkdir(parents=True, exist_ok=False)
    for row in rows:
        t2.prompt_ids(tokenizer, row, teacher=True)
        t2.prompt_ids(tokenizer, row)
    budget = t2.FidelityBudget()
    records = []
    cursor = 0
    stop = False

    def capture_and_persist(row):
        index = row["global_index"]
        record = t2.capture_one(index, row, endpoint=endpoint, model=SOURCE_MODEL, tokenizer=tokenizer, out=output)
        raw = (output / f"{index:04d}.json").read_bytes()
        target = remote / f"{index:05d}.json"
        target.write_bytes(raw)
        if target.read_bytes() != raw:
            raise ValueError("Completed A5 raw record remote readback differs")
        receipt = {
            "path": str(target),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "application_writes": 1,
        }
        (output / f"persist-{index:05d}.json").write_text(json.dumps(receipt) + "\n")
        return record

    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = set()
        while cursor < len(rows) or pending:
            while not stop and cursor < len(rows) and len(pending) < 8:
                row = rows[cursor]
                pending.add(pool.submit(capture_and_persist, row))
                cursor += 1
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                record = future.result()
                records.append(record)
                stop = budget.observe(record) or stop
    stop = budget.finish() or stop
    report = dict(
        planned=len(rows),
        attempted=len(records),
        unattempted=len(rows) - len(records),
        fidelity_rejects=budget.rejects,
        fidelity_checkpoints=budget.checkpoints,
        fatal_protocol=budget.fatal_protocol,
        stopped=stop,
        one_sample_per_original=True,
        application_retries=0,
        pool_and_training_admission=False,
        completed_record_stream=(
            "Exact raw bytes uploaded and full-readback verified before future completes; " "per-record local receipt"
        ),
    )
    (output / "capture-report.json").write_text(json.dumps(report, indent=2) + "\n")
    if stop:
        raise RuntimeError("A5 shard stopped on protocol failure or >5percent fidelity; no retry")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--shard", type=int, choices=range(4), required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()
    rows = checked_manifest(args.manifest, args.manifest_sha256, args.shard)
    capture(
        rows,
        endpoint=os.environ["PATH_B_A5_ENDPOINT"],
        tokenizer=t2.checked_tokenizer(args.tokenizer_dir),
        output=args.output,
        artifact_root=args.artifact_root,
    )


if __name__ == "__main__":
    main()
