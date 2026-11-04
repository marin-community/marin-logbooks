# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect only observer-confirmed terminal subset builds, retaining every attempt."""

import argparse
import concurrent.futures
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

import wandb
import zstandard
from google.cloud import storage

BASE = Path("logbooks/20-ema-gram-polar")
SCRATCH = Path("scratch")


def collect(index, cell, observed, manifest):
    output = SCRATCH / f"20261104-subset-hq01-build-{index:02d}-validated.json"
    if output.exists():
        return json.loads(output.read_text())
    assert observed["terminal"]
    job = observed["latest"]["job"]
    assert job["state"] == "JOB_STATE_SUCCEEDED" and job["exit_code"] == 0, job
    task_path = SCRATCH / f"20261104-subset-hq01-build-{index:02d}-task.json"
    if not task_path.exists():
        raw = subprocess.check_output(
            [
                "uv",
                "run",
                "--no-sync",
                "iris",
                "--cluster=marin",
                "rpc",
                "controller",
                "get-task-status",
                "--task-id",
                cell["job_id"] + "/0",
            ],
            text=True,
        )
        task_path.write_text(raw)
    task = json.loads(task_path.read_text())["task"]
    assert task["state"] == "TASK_STATE_SUCCEEDED" and task["exit_code"] == 0
    attempt = next(a for a in task["attempts"] if a["attempt_id"] == task["current_attempt_id"])
    assert attempt["state"] == "TASK_STATE_SUCCEEDED" and attempt["exit_code"] == 0
    archive = attempt["output_archive"]
    assert archive["state"] == "TASK_OUTPUT_ARCHIVE_STATE_UPLOADED"
    uri = archive["uri"]
    bucket, name = uri.removeprefix("gs://").split("/", 1)
    assert attempt["attempt_uid"] in name
    blob = storage.Client().bucket(bucket).get_blob(name)
    assert blob is not None and blob.size < 2_000_000
    raw = blob.download_as_bytes(if_generation_match=int(blob.generation), checksum="crc32c")
    assert hashlib.sha256(raw).hexdigest() == archive["sha256"]
    reader = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw))
    report = None
    with tarfile.open(fileobj=reader, mode="r|") as tar:
        for member in tar:
            if member.name.endswith("subset-report.json"):
                assert member.size < 1_000_000
                report = json.loads(tar.extractfile(member).read())
    assert report and report["success"] and report["source_commit"] == cell["source_commit"]
    assert report["task"] == manifest["tasks"][index]
    assert report["manifest_sha256"] == hashlib.sha256((BASE / "subset-inputs.json").read_bytes()).hexdigest()
    assert (
        report["source_sha256"] == hashlib.sha256(Path("experiments/ema_gram/subset_data.py").read_bytes()).hexdigest()
    )
    assert report["tokenizer_files"] == manifest["tokenizer_files"]
    assert report["result"]["full_native_readback"]
    assert report["result"]["rows"] > 0
    expected = sum(s["target_tokens"] for s in report["task"]["sources"])
    assert report["result"]["tokens"] >= expected
    api = wandb.Api()
    run = api.run("marin-community/marin/" + report["identity"])
    assert run.state == "finished" and run.summary["subset/success"] == 1
    assert run.config["source_commit"] == cell["source_commit"]
    assert run.config["manifest_sha256"] == report["manifest_sha256"]
    # Match logging IDs to the exact successful attempt in the owned log stream.
    lines = Path(observed["follow_log"]).read_text().splitlines()
    ids = set()
    for line in lines:
        if f"attempt={attempt['attempt_id']} |" in line:
            ids.update(re.findall(r"ema-gram-subset-logging-check-[a-f0-9]{12}", line))
    assert len(ids) == 1, ids
    logging_id = ids.pop()
    logging = api.run("marin-community/marin/" + logging_id)
    assert logging.state == "finished" and logging.summary["subset/success"] == 1
    assert logging.config["source_commit"] == cell["source_commit"]
    assert logging.config["manifest_sha256"] == report["manifest_sha256"]
    report.update(
        canonical_success=True,
        wandb_finished=True,
        canonical=job,
        task_status=task,
        artifact=dict(uri=uri, generation=blob.generation, crc32c=blob.crc32c, sha256=archive["sha256"]),
        online_runs=[dict(id=r.id, state=r.state, url=r.url) for r in (logging, run)],
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(dict(index=index, rows=report["result"]["rows"], tokens=report["result"]["tokens"], verified=True)),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((BASE / "subset-inputs.json").read_text())
    ledger = json.loads((SCRATCH / "20261104-subset-ledger.json").read_text())["jobs"]
    observers = json.loads((SCRATCH / "20261104-subset-hq01-build-observers.json").read_text())
    cells = {v["index"]: v for k, v in ledger.items() if "-hq01-build-" in k}
    selected = [i for i in cells if observers.get(str(i), {}).get("terminal")]
    if args.require_all:
        assert len(selected) == len(manifest["tasks"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(collect, i, cells[i], observers[str(i)], manifest) for i in selected]
        for future in futures:
            future.result()
    results = []
    for i in range(len(manifest["tasks"])):
        path = SCRATCH / f"20261104-subset-hq01-build-{i:02d}-validated.json"
        if not path.exists():
            return
        results.append(json.loads(path.read_text()))
    (BASE / "subset-build-results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("All build reports verified and ordered", flush=True)


if __name__ == "__main__":
    main()
