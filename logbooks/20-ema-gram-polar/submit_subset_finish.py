# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit each regional subset assembly/loader qualification stage exactly once."""

import argparse
import fcntl
import hashlib
import json
import shlex
import subprocess
from datetime import UTC, datetime

from submit_subset import BASE, LEDGER, ROOT, command, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("assemble", "data-check"), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.source_commit
    manifest_sha = hashlib.sha256((BASE / "subset-inputs.json").read_bytes()).hexdigest()
    if args.mode == "assemble":
        results = json.loads((BASE / "subset-build-results.json").read_text())
        manifest = json.loads((BASE / "subset-inputs.json").read_text())
        assert len(results) == len(manifest["tasks"])
        for item, task in zip(results, manifest["tasks"], strict=True):
            assert item["success"] and item["canonical_success"] and item["wandb_finished"]
            assert item["manifest_sha256"] == manifest_sha and item["task"] == task
            assert item["result"]["full_native_readback"]
        module = "experiments.ema_gram.subset_data"
        extra = ["--results", "logbooks/20-ema-gram-polar/subset-build-results.json"]
    else:
        dataset = json.loads((BASE / "subset-dataset.json").read_text())
        assert dataset["success"] and dataset["canonical_success"] and dataset["wandb_finished"]
        assert dataset["manifest_sha256"] == manifest_sha
        validated = json.loads((BASE / "subset-training-cpu-validation.json").read_text())
        assert validated["passed"]
        for path, sha in validated["source_files"].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha
        module = "experiments.ema_gram.subset_campaign"
        extra = []
    job = f"ema-gram-20-subset-hq01-{args.mode}-20261104"
    _, argv = command("probe", 0, args.source_commit)
    argv[argv.index("--job-name") + 1] = job
    if args.mode == "data-check":
        argv[argv.index("--cpu") + 1] = "8"
        argv[argv.index("--memory") + 1] = "32GB"
    logging = shlex.join(["python", "-m", module, "--mode", "logging-check", "--output", "/tmp/finish-logging.json"])
    worker = shlex.join(["python", "-m", module, "--mode", args.mode, *extra])
    argv[-1] = (
        'export PYTHONPATH="$PWD/lib/levanter/src:$PWD/lib/marin/src:$PWD"\n'
        "export JAX_PLATFORMS=cpu TOKENIZERS_PARALLELISM=false\n"
        f'{logging} && exec {worker} --output "${{IRIS_OUTPUT_DIR:-/tmp}}/subset-report.json"'
    )
    if not args.launch:
        print(json.dumps(dict(job=job, argv=argv)))
        return
    assert (
        subprocess.check_output(
            ["git", "ls-remote", "logbooks", "refs/heads/codex/20-ema-gram-polar"], text=True
        ).split()[0]
        == args.source_commit
    )
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    with LEDGER.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(LEDGER.read_text())
        assert job not in ledger["jobs"], "Stage already guarded; reconcile identity without resubmitting"
        record = dict(
            job_id=f"/kaiyuew@stanford.edu/{job}",
            mode=args.mode,
            index=0,
            source_commit=args.source_commit,
            launch_started=True,
            launched=False,
            submitted_utc=datetime.now(UTC).isoformat(),
            argv=argv,
        )
        ledger["jobs"][job] = record
        save(ledger)
        log = ROOT / "scratch" / f"{job}-submit.log"
        with log.open("w") as handle:
            result = subprocess.run(argv, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        record.update(launched=result.returncode == 0, submit_exit_code=result.returncode, submit_log=str(log))
        save(ledger)
        print(json.dumps(record))
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
