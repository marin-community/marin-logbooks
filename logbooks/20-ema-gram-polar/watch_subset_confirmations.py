# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Observe one accepted seed0 confirmation campaign without submitting jobs."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from adopt_alpha0_long_run import accounting
from prepare_parallel_long_runs import accounting as parallel_accounting
from prepare_parallel_long_runs import resolve_bundle
from prepare_subset_confirmations import prepare
from submit_subset_confirmations import command, validate_completions
from watch_subset_expansions import observe_campaign

from experiments.ema_gram.plan import read_lock


def accepted_entries(
    ledger, binding, campaign, bundle, bundle_path, lock_path, amendment_path=None, completions_path=None
):
    """Join accepted submissions to exact frozen configurations and source."""
    if ledger["binding"] != binding:
        raise ValueError("Confirmation ledger binding changed")
    accepted = {}
    for key, row in ledger["cells"].items():
        index = int(key)
        if str(index) != key or not 0 <= index < len(bundle["cells"]):
            raise ValueError("Unexpected confirmation index")
        expected = bundle["cells"][index]
        job_id = f"/kaiyuew@stanford.edu/{campaign}-{index:02d}"
        argv = command(
            index,
            expected["variant"],
            binding["source"],
            campaign,
            bundle_path,
            lock_path,
            amendment_path,
            completions_path,
        )
        if row["job_id"] != job_id or row["argv"] != argv or not row["launch_started"]:
            raise ValueError("Submission differs from frozen confirmation identity")
        if row.get("launched") and row.get("submit_exit_code") == 0:
            accepted[key] = dict(job_id=job_id, source_commit=binding["source"], config_sha256=expected["config_sha256"])
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worktree", "bundle", "completions"):
        parser.add_argument("--" + name, type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--lock", type=Path)
    selection.add_argument("--amendment", type=Path)
    parser.add_argument("--adopt-baseline", type=Path)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch("ema-gram-20-confirm-seed0-[a-z0-9-]+", args.campaign):
        raise ValueError("Invalid confirmation campaign")
    worktree = args.worktree.resolve()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip()
    if source != args.source_commit:
        raise ValueError("Execution worktree source changed")
    inputs = {}
    paths = [args.bundle, args.lock if args.amendment is None else args.amendment, args.completions]
    if args.adopt_baseline is not None:
        paths.append(args.adopt_baseline)
    for path in paths:
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Inputs must be tracked relative paths")
        committed = subprocess.check_output(["git", "show", f"{source}:{path}"], cwd=worktree)
        if committed != (worktree / path).read_bytes():
            raise ValueError("Confirmation input differs from committed source")
        inputs[str(path)] = hashlib.sha256(committed).hexdigest()
    binding = dict(source=source, inputs=inputs)
    if args.amendment is None:
        lock = read_lock(worktree / args.lock)
        bundle = json.loads((worktree / args.bundle).read_text())
        if bundle != prepare(lock, bundle["artifact_version"]):
            raise ValueError("Frozen confirmation bundle differs from selection lock")
        validate_completions(lock, json.loads((worktree / args.completions).read_text()))
        allocations = accounting(worktree, args.adopt_baseline, lock, bundle)
    else:
        bundle = resolve_bundle(worktree / args.bundle, worktree / args.completions, worktree / args.amendment)
        allocations = parallel_accounting(worktree, args.adopt_baseline, bundle, args.amendment)
    launch_indexes = {str(row["index"]) for row in allocations if row["disposition"] == "submit"}
    ledger_path = worktree / "scratch" / f"{args.campaign}-ledger.json"

    def read_accepted():
        ledger = json.loads(ledger_path.read_text())
        if set(ledger["cells"]) - launch_indexes:
            raise ValueError("Adopted baseline must retain its existing observer")
        if ledger.get("confirmation_accounting") != allocations:
            raise ValueError("Final confirmation accounting differs from submission")
        return accepted_entries(
            ledger, binding, args.campaign, bundle, args.bundle, args.lock, args.amendment, args.completions
        )

    read_accepted()  # Reject absent or mismatched submission evidence before creating observer state.
    observe_campaign(worktree, args.campaign, binding, len(launch_indexes), read_accepted)


if __name__ == "__main__":
    main()
