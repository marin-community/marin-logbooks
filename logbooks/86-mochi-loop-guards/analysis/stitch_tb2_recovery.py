# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select one registered replay for originally ungraded TB2 task/repetition identities."""

import argparse
import gzip
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import yaml
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from skyrl_train.mismatch_probe.protocol import request_seed

from vendor.skyrl.scripts.loop_audit import is_infrastructure_error, read_rows


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def row_key(row):
    return (
        row["uid"],
        digest(row["response_token_ids"]),
        digest(row.get("messages")),
        (row.get("verification") or {}).get("status"),
        row.get("exception_type"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run-output", required=True)
    parser.add_argument("--recovery-run-output", required=True)
    parser.add_argument("--recovery-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    receipt = json.loads(StoragePath(args.recovery_receipt).read_text())
    ledger = json.loads(StoragePath(receipt["original_ledger"]).read_text())
    original_rows = list(read_rows(ledger["source"]))
    original_groups = defaultdict(list)
    for row in original_rows:
        original_groups[row_key(row)].append(row)
    originals = {}
    for item in ledger["trials"]:
        group = original_groups[tuple(item["join_key"])]
        if len(group) != 1:
            raise ValueError("Original row/repetition attribution is ambiguous")
        identity = item["task_id"], item["repetition_id"]
        originals[identity] = (group[0], item)
    targets = {(item["task_id"], item["repetition_id"]) for item in receipt["target_trials"]}
    task_ids = set(receipt["selected_task_ids"])
    if len(originals) != 267 or len(targets) != 33 or len(task_ids) != 18:
        raise ValueError("Recovery census differs from the registered rule")
    if {identity for identity, (row, _) in originals.items() if is_infrastructure_error(row)} != targets:
        raise ValueError("Recovery targets include a graded outcome or omit an original infrastructure outcome")

    config_path = "resolved-launch.yaml"
    original_config = yaml.safe_load((StoragePath(args.original_run_output) / config_path).read_text())["config"]
    recovery_root = StoragePath(args.recovery_run_output)
    recovery_config = yaml.safe_load((recovery_root / config_path).read_text())["config"]
    original_model = original_config["skyrl"]["trainer"]["policy"]["model"]
    recovery_model = recovery_config["skyrl"]["trainer"]["policy"]["model"]
    if original_model != recovery_model:
        raise ValueError("Recovery changed the initializer")
    paths = list((recovery_root / "exports/dumped_evals/*/eval_only/DCAgent2_terminal_bench_2.jsonl").glob())
    if len(paths) != 1:
        raise ValueError("Select exactly one completed recovery evaluation session")
    replay_rows = [row for row in read_rows(str(paths[0])) if row["task_id"] in task_ids]
    if len(replay_rows) != 54:
        raise ValueError("Expected all 18 recovery tasks at all three frozen repetitions")
    replay_groups = defaultdict(list)
    task_by_uid = {}
    for row in replay_rows:
        replay_groups[row_key(row)].append(row)
        task_by_uid[row["uid"]] = row["task_id"]
    trajectory_root = StoragePath(recovery_config["artifacts"]["attempts_root"]) / "trajectories"

    def extract(path):
        extracted = []
        with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
            for name in archive.namelist():
                if not name.startswith("records/") or not name.endswith(".json.gz"):
                    continue
                record = json.loads(gzip.decompress(archive.read(name)))
                if record["trajectory"]["environment_extras"].get("data_source") != "DCAgent2/terminal_bench_2":
                    continue
                uid = record["trajectory"]["instance_id"].rsplit("-", 1)[-1]
                if uid not in task_by_uid:
                    continue
                key = (
                    uid,
                    digest(record["response"]["token_ids"]),
                    digest(record["response"].get("messages")),
                    (record.get("verification_result") or {}).get("status"),
                    record["disposition"].get("exception_type"),
                )
                extracted.append(
                    (
                        key,
                        record["trajectory"]["repetition_id"],
                        record["provenance"]["sampling"].get("seed"),
                        record["record_id"],
                        str(path),
                    )
                )
        return extracted

    archive_paths = sorted((trajectory_root / "schema_v7/archives/phase=eval/step=eval_only/*.zip").glob(), key=str)
    retained = defaultdict(list)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for records in pool.map(extract, archive_paths):
            for key, repetition, seed, record_id, archive in records:
                retained[key].append((repetition, seed, record_id, archive))
    if {key: len(value) for key, value in retained.items()} != {key: len(value) for key, value in replay_groups.items()}:
        raise ValueError("Retained/replay scientific-content multisets differ")
    replays = {}
    for key, records in retained.items():
        if len(records) != 1:
            raise ValueError("Replay row/repetition attribution is ambiguous; retain richer identity before stitching")
        repetition, seed, record_id, archive = records[0]
        row = replay_groups[key][0]
        identity = row["task_id"], repetition
        if identity in replays or seed != request_seed(20261125, identity[0], identity[1]):
            raise ValueError("Repeated identity or changed frozen seed")
        replays[identity] = (row, record_id, archive)
    if set(replays) != {(task_id, repetition) for task_id in task_ids for repetition in range(3)}:
        raise ValueError("Recovery task/repetition set is incomplete")

    result = []
    replacements = []
    for identity, (original, metadata) in sorted(originals.items()):
        row = dict(original)
        provenance = {
            "task_id": identity[0],
            "repetition_id": identity[1],
            "sampling_seed": request_seed(20261125, *identity),
            "original_record_id": metadata["record_id"],
            "original_dump_ordinal": original["ordinal"],
        }
        if identity in targets:
            replay, record_id, archive = replays[identity]
            row = dict(replay)
            provenance.update(replaced_original_infrastructure=True, replay_record_id=record_id, replay_archive=archive)
            replacements.append(
                {
                    **provenance,
                    "replay_verification_status": (replay.get("verification") or {}).get("status"),
                    "replay_still_infrastructure": is_infrastructure_error(replay),
                }
            )
        elif {key: row[key] for key in original} != original:
            raise ValueError("An original graded outcome changed")
        row["recovery_provenance"] = provenance
        result.append(row)
    destination = StoragePath(args.output)
    destination.mkdirs()
    (destination / "DCAgent2_terminal_bench_2.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result)
    )
    report = {
        "analysis_source_sha256": args.source_sha256,
        "original_run_output": args.original_run_output,
        "recovery_run_output": args.recovery_run_output,
        "registered_recovery_receipt": args.recovery_receipt,
        "trials": len(result),
        "original_graded_preserved": len(result) - len(replacements),
        "original_infrastructure_replaced": len(replacements),
        "extra_replayed_graded_ignored": len(replays) - len(replacements),
        "remaining_infrastructure": sum(is_infrastructure_error(row) for row in result),
        "frozen_task_repetition_seed_set_sha256": digest(
            [(task, repetition, request_seed(20261125, task, repetition)) for task, repetition in sorted(originals)]
        ),
        "replacements": replacements,
    }
    (destination / "stitch-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "replacements"}))


if __name__ == "__main__":
    main()
