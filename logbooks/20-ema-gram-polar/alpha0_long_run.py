# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare, execute and observe the settled alpha0 confirmation independently."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import draccus
from levanter.main.train_lm import TrainLmConfig
from levanter.tracker.wandb import WandbConfig
from prepare_alpha0_long_run import prepare
from submit_subset_confirmations import claim, persist
from submit_subset_training import command as pilot_command
from watch_subset_expansions import observe_campaign

from experiments.ema_gram.plan import SELECTION_METRIC
from experiments.ema_gram.subset_campaign import prepared
from experiments.ema_gram.train import run_training

BASE = Path(__file__).parent
ROOT = BASE.parents[1]


def resolve(bundle_path, completions_path):
    """Replay baseline selection and decode the exact native long-run config."""
    bundle = json.loads(bundle_path.read_text())
    records = json.loads(completions_path.read_text())
    if bundle != prepare(records, bundle["artifact_version"]):
        raise ValueError("Baseline bundle differs from verified pilot selection")
    row = bundle["cells"][0]
    encoded = row["config"]
    config = draccus.decode(TrainLmConfig, encoded["train_config"])
    config = replace(
        config,
        trainer=replace(
            config.trainer,
            jax_config=encoded["train_config"]["trainer"]["jax_config"],
            load_checkpoint_path=encoded["train_config"]["trainer"]["load_checkpoint_path"],
        ),
    )
    pod = replace(
        prepared(0, row["variant"]),
        train_config=config,
        output_path=encoded["output_path"],
        env_vars=encoded["env_vars"],
        auto_build_caches=False,
    )
    if draccus.encode(pod) != encoded:
        raise ValueError("Native baseline config changed during decoding")
    return pod, bundle


def validate_protocol(protocol, bundle):
    """Require a recorded early-start amendment and predeclared common target."""
    if protocol["status"] != "frozen" or protocol["scope"] != "settled_alpha0_seed0":
        raise ValueError("Early-start protocol is still a draft")
    if protocol["config_sha256"] != bundle["cells"][0]["config_sha256"]:
        raise ValueError("Early-start protocol belongs to another configuration")
    if protocol["predecessor_records_sha256"] != bundle["predecessor_records_sha256"]:
        raise ValueError("Early-start protocol belongs to another pilot evidence set")
    target = protocol["common_loss_target"]
    if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target) or target <= 0:
        raise ValueError("A finite positive common loss target must be declared before execution")
    if protocol["selection_metric"] != SELECTION_METRIC or not protocol["target_rationale"].strip():
        raise ValueError("Missing common target metric or rationale")
    if protocol["confirmation_outcomes_used_for_selection"] is not False:
        raise ValueError("Confirmation outcomes must not influence remaining pilot selection")
    if protocol["global_selection_locked"] is not False or protocol["allow_settled_alpha0_early_start"] is not True:
        raise ValueError("Record the limited early-start amendment without claiming a global lock")
    if protocol["baseline_reuse_policy"] != "adopt_exact_config_once_in_final_confirmation_set":
        raise ValueError("Future confirmations must reconcile this baseline instead of duplicating it")
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def command(source, campaign, bundle, completions, protocol):
    argv = pilot_command(0, "v4-8", source)
    argv[argv.index("--job-name") + 1] = campaign + "-00"
    common = [
        "python",
        "logbooks/20-ema-gram-polar/alpha0_long_run.py",
        "--bundle",
        str(bundle),
        "--completions",
        str(completions),
        "--protocol",
        str(protocol),
    ]
    gate = (
        "python logbooks/20-ema-gram-polar/check_campaign_caches.py "
        "--manifest logbooks/20-ema-gram-polar/subset-cache-manifest.json "
        "--output /tmp/alpha0-cache-check.json"
    )
    logging = shlex.join([*common, "--mode", "logging-check", "--output", "/tmp/alpha0-host-logging.json"])
    worker = shlex.join([*common, "--mode", "worker", "--output", "/tmp/alpha0-worker.json"])
    argv[-1] = (
        'export PYTHONPATH="$PWD/logbooks/20-ema-gram-polar:$PWD/lib/levanter/src:$PWD/lib/marin/src:$PWD"\n'
        + f"{gate} && JAX_PLATFORMS=cpu {logging} && exec {worker}"
    )
    return argv


def execute(args, pod, bundle, protocol_digest):
    record = dict(
        config=bundle["cells"][0]["config"],
        config_sha256=bundle["cells"][0]["config_sha256"],
        protocol_digest=protocol_digest,
        bundle_sha256=hashlib.sha256((args.worktree / args.bundle).read_bytes()).hexdigest(),
        source_commit=os.environ.get("GIT_COMMIT"),
        task=os.environ.get("IRIS_TASK_ID"),
        attempt_uid=os.environ.get("IRIS_ATTEMPT_UID"),
        entrypoint_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    if args.mode == "worker":
        run_training(pod)
        record["native_returned"] = True
    else:
        identity = "ema-gram-alpha0-long-host-" + uuid.uuid4().hex[:12]
        tracker = WandbConfig(
            entity="marin-community",
            project="marin",
            id=identity,
            name=identity,
            group="ema-gram-20-alpha0-long",
            mode="online",
            resume="never",
            background=False,
            save_code=False,
        ).init(None)
        print(f"W&B: https://wandb.ai/marin-community/marin/runs/{identity}", flush=True)
        try:
            tracker.log_hyperparameters(record)
            tracker.log({"logging_check/success": 1}, step=0)
            record.update(identity=identity, success=True)
        finally:
            tracker.finish()
    args.output.write_text(json.dumps(record, indent=2) + "\n")


def committed_binding(worktree, source, paths):
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip() != source:
        raise ValueError("Execution source changed")
    inputs = {}
    for path in paths:
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Use tracked relative input paths")
        committed = subprocess.check_output(["git", "show", f"{source}:{path}"], cwd=worktree)
        if committed != (worktree / path).read_bytes():
            raise ValueError("Frozen input differs from committed source")
        inputs[str(path)] = hashlib.sha256(committed).hexdigest()
    return dict(source=source, inputs=inputs)


def accepted_entries(ledger, binding, campaign, argv, bundle):
    if ledger["binding"] != binding:
        raise ValueError("Baseline ledger binding changed")
    accepted = {}
    for index, row in ledger["cells"].items():
        job = f"/kaiyuew@stanford.edu/{campaign}-00"
        if index != "0" or row["job_id"] != job or row["argv"] != argv or not row["launch_started"]:
            raise ValueError("Baseline submission differs from frozen identity")
        if row.get("launched") and row.get("submit_exit_code") == 0:
            accepted[index] = dict(
                job_id=job, source_commit=binding["source"], config_sha256=bundle["cells"][0]["config_sha256"]
            )
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "completions", "protocol"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--mode", choices=("plan", "submit", "observe", "logging-check", "worker"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worktree", type=Path, default=ROOT)
    parser.add_argument("--source-commit")
    parser.add_argument("--published-ref")
    parser.add_argument("--campaign")
    args = parser.parse_args()
    worktree = args.worktree.resolve()
    pod, bundle = resolve(worktree / args.bundle, worktree / args.completions)
    protocol = json.loads((worktree / args.protocol).read_text())
    if args.mode in ("logging-check", "worker"):
        if args.output is None:
            raise ValueError("Execution requires an output record path")
        digest = validate_protocol(protocol, bundle)
        execute(args, pod, bundle, digest)
        return
    if not args.campaign or not re.fullmatch("ema-gram-20-alpha0-seed0-[a-z0-9-]+", args.campaign):
        raise ValueError("Use a distinct alpha0 campaign identity")
    if not args.source_commit or not re.fullmatch("[0-9a-f]{40}", args.source_commit):
        raise ValueError("Exact execution source is required")
    argv = command(args.source_commit, args.campaign, args.bundle, args.completions, args.protocol)
    if args.mode == "plan":
        print(json.dumps(dict(submitted=False, commands=[argv], protocol_status=protocol["status"])))
        return
    validate_protocol(protocol, bundle)
    binding = committed_binding(worktree, args.source_commit, (args.bundle, args.completions, args.protocol))
    ledger_path = worktree / "scratch" / f"{args.campaign}-ledger.json"
    if args.mode == "observe":

        def read_accepted():
            return accepted_entries(json.loads(ledger_path.read_text()), binding, args.campaign, argv, bundle)

        if not read_accepted():
            raise ValueError("No accepted baseline submission to observe")
        observe_campaign(worktree, args.campaign, binding, 1, read_accepted)
        return
    if (
        not args.published_ref
        or subprocess.check_output(
            ["git", "ls-remote", "logbooks", args.published_ref], cwd=worktree, text=True
        ).split()[0]
        != args.source_commit
    ):
        raise ValueError("Publish exact execution source before submitting")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=worktree):
        raise ValueError("Execution worktree has tracked changes")
    qualification = json.loads(
        (worktree / "logbooks/20-ema-gram-polar/alpha0-long-run-runtime-preflight.json").read_text()
    )
    worker_path = worktree / "logbooks/20-ema-gram-polar/alpha0_long_run.py"
    if not qualification["passed"] or not qualification["logging_check"]["success"]:
        raise ValueError("Baseline worker qualification is incomplete")
    if qualification["source_sha256"] != hashlib.sha256(worker_path.read_bytes()).hexdigest():
        raise ValueError("Qualified baseline worker source changed")
    ledger_path.parent.mkdir(exist_ok=True)
    with ledger_path.with_suffix(".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(binding=binding, cells={})
        accepted_entries(ledger, binding, args.campaign, argv, bundle)
        if ledger["cells"]:
            raise ValueError("Baseline identity already guarded; reconcile without resubmission")
        subprocess.run(
            [
                sys.executable,
                str(worktree / "logbooks/20-ema-gram-polar/check_campaign_caches.py"),
                "--full",
                "--manifest",
                str(worktree / "logbooks/20-ema-gram-polar/subset-cache-manifest.json"),
                "--output",
                str(ledger_path.with_suffix(".cache-check.json")),
            ],
            cwd=worktree,
            check=True,
        )
        claim(
            ledger,
            0,
            dict(
                job_id=f"/kaiyuew@stanford.edu/{args.campaign}-00",
                argv=argv,
                submitted_utc=datetime.now(UTC).isoformat(),
            ),
        )
        persist(ledger_path, ledger)
        output = ledger_path.with_suffix(".submit.log")
        with output.open("w") as stream:
            result = subprocess.run(argv, cwd=worktree, stdout=stream, stderr=subprocess.STDOUT)
        ledger["cells"]["0"].update(
            submit_exit_code=result.returncode, launched=result.returncode == 0, submit_log=str(output)
        )
        persist(ledger_path, ledger)
        if result.returncode:
            raise RuntimeError("Submission error: preserve guard and reconcile canonical identity")


if __name__ == "__main__":
    main()
