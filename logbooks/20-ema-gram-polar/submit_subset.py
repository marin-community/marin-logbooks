# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the frozen subset CPU stages once, preserving ambiguous submissions."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).parent
LEDGER = ROOT / "scratch/20261104-subset-ledger.json"
MUTEX = threading.Lock()


def command(mode, index, source):
    label = "probe" if mode == "probe" else f"build-{index:02d}"
    job = f"ema-gram-20-subset-{label}-20261104"
    common = ["python", "-m", "experiments.ema_gram.subset_data"]
    logging = shlex.join([*common, "--mode", "logging-check", "--output", "/tmp/subset-logging.json"])
    worker = shlex.join([*common, "--mode", mode, "--index", str(index)])
    shell = 'export PYTHONPATH="$PWD/lib/levanter/src:$PWD/lib/marin/src:$PWD"\n'
    shell += "export JAX_PLATFORMS=cpu TOKENIZERS_PARALLELISM=false\n"
    shell += f'{logging} && exec {worker} --output "${{IRIS_OUTPUT_DIR:-/tmp}}/subset-report.json"'
    return job, [
        "uv",
        "run",
        "--no-sync",
        "iris",
        "--cluster=marin",
        "job",
        "run",
        "--no-wait",
        "--no-cancel-on-exit",
        "--job-name",
        job,
        "--priority",
        "batch",
        "--enable-extra-resources",
        "--cpu",
        "2",
        "--memory",
        "8GB",
        "--disk",
        "20GB",
        "--region",
        "us-central2",
        "--sync-package",
        "marin-core",
        "--sync-package",
        "marin-levanter",
        "--sync-package",
        "marin-dupekit",
        "-e",
        "GIT_COMMIT",
        source,
        "--",
        "bash",
        "-c",
        shell,
    ]


def save(ledger):
    temp = LEDGER.with_suffix(".tmp")
    with temp.open("w") as handle:
        handle.write(json.dumps(ledger, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(LEDGER)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--mode", choices=("probe", "build"), required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.source_commit
    validation = json.loads((BASE / "subset-preflight.json").read_text())
    assert validation["passed"]
    for path, sha in validation["source_files"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha, path
    manifest = json.loads((BASE / "subset-inputs.json").read_text())
    assert manifest["tokenizer_files"] == validation["tokenizer_files"]
    if args.mode == "build":
        probe = json.loads((BASE / "subset-regional-probe.json").read_text())
        assert probe["success"] and probe["canonical_success"] and probe["wandb_finished"]
        assert probe["manifest_sha256"] == hashlib.sha256((BASE / "subset-inputs.json").read_bytes()).hexdigest()
        assert len(probe["sources"]) == 7
    if args.launch:
        assert (
            subprocess.check_output(
                ["git", "ls-remote", "logbooks", "refs/heads/codex/20-ema-gram-polar"], text=True
            ).split()[0]
            == args.source_commit
        )
        assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    indices = [0] if args.mode == "probe" else range(len(manifest["tasks"]))
    LEDGER.parent.mkdir(exist_ok=True)
    with LEDGER.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"jobs": {}}

        def submit(index):
            job, argv = command(args.mode, index, args.source_commit)
            with MUTEX:
                if job in ledger["jobs"]:
                    print(f"Already guarded: {job}", flush=True)
                    return
                if not args.launch:
                    print(json.dumps(dict(job=job, argv=argv)), flush=True)
                    return
                record = dict(
                    job_id=f"/kaiyuew@stanford.edu/{job}",
                    mode=args.mode,
                    index=index,
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
            with MUTEX:
                record.update(launched=result.returncode == 0, submit_exit_code=result.returncode, submit_log=str(log))
                save(ledger)
            print(json.dumps(record), flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(submit, indices))


if __name__ == "__main__":
    main()
