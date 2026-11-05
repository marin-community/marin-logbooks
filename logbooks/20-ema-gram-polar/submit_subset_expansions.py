# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit complete-group expansions once through a shared permanent launch registry."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

from prepare_subset_expansions import prepare
from submit_subset_confirmations import persist
from submit_subset_training import command as pilot_command

from experiments.ema_gram.plan import Cell

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "logbooks/20-ema-gram-polar"


def command(index, variant, source, campaign, bundle, completions, accepted):
    argv = pilot_command(index, variant, source)
    argv[argv.index("--job-name") + 1] = f"{campaign}-{index:02d}"
    common = [
        "python",
        "logbooks/20-ema-gram-polar/expansion_worker.py",
        "--bundle",
        str(bundle),
        "--completions",
        str(completions),
        "--accepted-expansions",
        str(accepted),
        "--index",
        str(index),
    ]
    logging = shlex.join([*common, "--mode", "logging-check", "--output", "/tmp/expansion-host-logging.json"])
    worker = shlex.join([*common, "--mode", "worker", "--output", "/tmp/expansion-worker.json"])
    gate = (
        "python logbooks/20-ema-gram-polar/check_campaign_caches.py "
        "--manifest logbooks/20-ema-gram-polar/subset-cache-manifest.json "
        "--output /tmp/expansion-cache-check.json"
    )
    argv[-1] = (
        'export PYTHONPATH="$PWD/logbooks/20-ema-gram-polar:$PWD/lib/levanter/src:$PWD/lib/marin/src:$PWD"\n'
        + f"{gate} && JAX_PLATFORMS=cpu {logging} && exec {worker}"
    )
    return argv


def claim(registry, cell, record):
    name = Cell(**cell).name
    if name in registry["cells"]:
        raise ValueError("Expansion already guarded across campaigns; reconcile its original identity")
    registry["cells"][name] = dict(record, cell=cell, launch_started=True, launched=False)


def persist_registry(path, registry):
    persist(path, registry)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_registry(path):
    """Fail closed if a previously initialized registry is missing."""
    marker = path.with_suffix(".initialized.json")
    if marker.exists():
        if not path.exists():
            raise ValueError("Permanent registry missing; reconcile preserved evidence before proceeding")
        return json.loads(path.read_text())
    if path.exists() or list(path.parent.glob("ema-gram-20-expand-seed0-*-submit.log")):
        raise ValueError("Existing expansion evidence lacks registry marker; explicit reconciliation required")
    with marker.open("x") as stream:
        stream.write(
            json.dumps(dict(registry=str(path.resolve()), initialized_utc=datetime.now(UTC).isoformat())) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    registry = dict(cells={}, campaigns={})
    persist_registry(path, registry)
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "completions", "accepted-expansions", "registry"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-commit", "published-ref", "campaign"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT.resolve():
        raise ValueError("Run from the immutable execution worktree root")
    if not re.fullmatch("[0-9a-f]{40}", args.source_commit):
        raise ValueError("Require an exact source commit")
    if not re.fullmatch("ema-gram-20-expand-seed0-[a-z0-9-]+", args.campaign):
        raise ValueError("Use a distinct expansion campaign identity")
    expected_registry = ROOT.parent / "ema-gram-research-20/scratch/20261105-subset-expansion-registry.json"
    if args.registry.resolve() != expected_registry.resolve():
        raise ValueError("All expansion campaigns must share the permanent research registry")
    paths = [args.bundle, args.completions, args.accepted_expansions]
    for path in paths:
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Inputs must be tracked within the immutable execution worktree")
        subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], check=True, stdout=subprocess.DEVNULL)
    bundle = json.loads(args.bundle.read_text())
    records = json.loads(args.completions.read_text())
    accepted = json.loads(args.accepted_expansions.read_text())
    if bundle != prepare(records, bundle["artifact_version"], accepted) or not bundle["cells"]:
        raise ValueError("Frozen bundle differs from required expansions, or no expansion is due")
    qualification = json.loads((BASE / "expansion-worker-preflight.json").read_text())
    if not qualification["passed"] or not qualification["logging_check"]["success"]:
        raise ValueError("Expansion worker qualification incomplete")
    if qualification["source_sha256"] != hashlib.sha256((BASE / "expansion_worker.py").read_bytes()).hexdigest():
        raise ValueError("Qualified expansion worker changed")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != args.source_commit:
        raise ValueError("Execution HEAD differs from exact source")
    commands = [
        command(
            i, row["variant"], args.source_commit, args.campaign, args.bundle, args.completions, args.accepted_expansions
        )
        for i, row in enumerate(bundle["cells"])
    ]
    if not args.launch:
        print(json.dumps(dict(submitted=False, commands=commands)))
        return
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("Publish tracked changes before launch")
    if (
        subprocess.check_output(["git", "ls-remote", "logbooks", args.published_ref], text=True).split()[0]
        != args.source_commit
    ):
        raise ValueError("Exact source is not published")
    args.registry.parent.mkdir(exist_ok=True)
    with args.registry.with_suffix(".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry = read_registry(args.registry)
        binding = dict(
            source=args.source_commit,
            artifact_version=bundle["artifact_version"],
            inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        )
        old = registry["campaigns"].get(args.campaign)
        if old is not None and old != binding:
            raise ValueError("Existing campaign is bound to different source/inputs")
        if any(
            name != args.campaign and value["artifact_version"] == bundle["artifact_version"]
            for name, value in registry["campaigns"].items()
        ):
            raise ValueError("Artifact version already belongs to another expansion campaign")
        names = [Cell(**r["cell"]).name for r in bundle["cells"]]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate expansion setting in frozen bundle")
        if any(name in registry["cells"] for name in names):
            raise ValueError("At least one proposed expansion is already guarded; no resubmission")
        subprocess.run(
            [
                sys.executable,
                str(BASE / "check_campaign_caches.py"),
                "--full",
                "--manifest",
                str(BASE / "subset-cache-manifest.json"),
                "--output",
                str(args.registry.with_suffix(".cache-check.json")),
            ],
            check=True,
        )
        registry["campaigns"][args.campaign] = binding
        mutex = threading.Lock()

        def submit(index):
            row = bundle["cells"][index]
            record = dict(
                campaign=args.campaign,
                index=index,
                source_commit=args.source_commit,
                job_id=f"/kaiyuew@stanford.edu/{args.campaign}-{index:02d}",
                argv=commands[index],
                config_sha256=row["config_sha256"],
                submitted_utc=datetime.now(UTC).isoformat(),
            )
            with mutex:
                claim(registry, row["cell"], record)
                persist_registry(args.registry, registry)
            output = args.registry.parent / f"{args.campaign}-{index:02d}-submit.log"
            with output.open("x") as stream:
                result = subprocess.run(commands[index], stdout=stream, stderr=subprocess.STDOUT)
            with mutex:
                registry["cells"][names[index]].update(
                    submit_exit_code=result.returncode, launched=result.returncode == 0, submit_log=str(output)
                )
                persist_registry(args.registry, registry)
            return result.returncode

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(commands)) as pool:
            codes = list(pool.map(submit, range(len(commands))))
        if any(codes):
            raise RuntimeError("Submission error: preserve all guards and reconcile exact identities")


if __name__ == "__main__":
    main()
