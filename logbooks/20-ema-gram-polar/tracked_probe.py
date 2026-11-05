# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the existing checkpoint probe with fresh ONLINE tracking before any kernels."""

import argparse
import hashlib
import json
import os
import runpy
import sys
import uuid
from pathlib import Path

import wandb
from levanter.tracker.wandb import WandbTracker
from rigging.secrets import require_secret

ROOT = Path(__file__).resolve().parents[2]


def source_files():
    paths = [
        Path(__file__).relative_to(ROOT),
        Path("experiments/ema_gram/probe.py"),
        Path("experiments/ema_gram/test_experiment.py"),
        Path("lib/levanter/src/levanter/optim/ema_gram.py"),
    ]
    return {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def resolve_config(path, index):
    row = json.loads(path.read_text())[index]
    row = row.get("config", row)
    config = row["train_config"]
    if config["optimizer"]["type"] != "ema_gram_polar":
        raise ValueError("An EMA-Gram configuration is required")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("logging-check", "cpu-check", "probe"), required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--config-index", type=int, default=0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--expected-counter", type=int)
    parser.add_argument("--dataset", default="uncheatable_eval/wikipedia_english")
    parser.add_argument("--batch-offset", type=int, default=0)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--step-sizes", nargs="+", type=float, default=[1e-5, 3e-5, 1e-4])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = resolve_config(args.configs, args.config_index)
    if args.mode == "probe":
        if not args.checkpoint or args.expected_counter not in (3733, 18665, 33597):
            raise ValueError("Identify one preregistered reference checkpoint and its native counter")
        if (config["optimizer"]["alpha"], config["trainer"]["seed"], config["trainer"]["num_train_steps"]) != (
            0.25,
            0,
            37335,
        ):
            raise ValueError("Probes require the seed0 alpha=.25 full-confirmation configuration")
    identity = "ema-gram-reference-probe-" + args.mode + "-" + uuid.uuid4().hex[:12]
    destination = args.output_dir / identity
    destination.mkdir(parents=True, exist_ok=False)
    run = wandb.init(
        entity="marin-community",
        project="marin",
        id=identity,
        name=identity,
        group="ema-gram-20-reference-probes",
        mode="online",
        resume="never",
        settings={"api_key": require_secret("WANDB_API_KEY")},
    )
    assert run.settings.mode == "online"
    tracker = WandbTracker(run, replicate_path=str(destination))
    metadata = dict(
        mode=args.mode,
        identity=identity,
        source_files=source_files(),
        source_commit=os.environ.get("GIT_COMMIT"),
        resolved_config=config,
        checkpoint=args.checkpoint,
        expected_counter=args.expected_counter,
        dataset=args.dataset,
        batch_offset=args.batch_offset,
        batches=args.batches,
        step_sizes=args.step_sizes,
        cpu_fixture_seeds=[1, 2, 3] if args.mode == "cpu-check" else None,
    )
    tracker.log_hyperparameters(metadata)
    (destination / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"W&B: {run.url}", flush=True)
    try:
        tracker.log({"logging_check/success": 1}, step=0)
        if args.mode == "cpu-check":
            import jax  # noqa: PLC0415 - initialize tracking before numerical imports

            from experiments.ema_gram.test_experiment import (  # noqa: PLC0415
                test_checkpoint_probes_match_each_matrix_and_keep_nonmatrices_fixed,
            )

            assert jax.default_backend() == "cpu"
            test_checkpoint_probes_match_each_matrix_and_keep_nonmatrices_fixed()
            tracker.log({"probe_cpu/norm_matching_and_paired_loss_oracle": 1}, step=1)
        elif args.mode == "probe":
            from iris.jax.init import initialize_jax  # noqa: PLC0415

            initialize_jax()
            normalized = destination / "resolved-configs.json"
            normalized.write_text(json.dumps([{"train_config": config}], indent=2) + "\n")
            report_path = destination / "report.json"
            saved_argv = sys.argv
            try:
                sys.argv = [
                    "probe",
                    "--configs",
                    str(normalized),
                    "--checkpoint",
                    args.checkpoint,
                    "--expected-counter",
                    str(args.expected_counter),
                    "--dataset",
                    args.dataset,
                    "--batch-offset",
                    str(args.batch_offset),
                    "--batches",
                    str(args.batches),
                    "--step-sizes",
                    *map(str, args.step_sizes),
                    "--output",
                    str(report_path),
                ]
                runpy.run_module("experiments.ema_gram.probe", run_name="__main__")
            finally:
                sys.argv = saved_argv
            report = json.loads(report_path.read_text())
            if report["step"] != args.expected_counter:
                raise ValueError("Restored native counter differs from reference checkpoint")
            with (destination / "loss-rows.jsonl").open("w") as stream:
                for step, row in enumerate(report["loss_probes"], 1):
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    tracker.log(
                        {
                            "probe/alpha": row["alpha"],
                            "probe/ridge": row["ridge"],
                            "probe/step_size": row["step_size"],
                            "probe/baseline_loss_mean": sum(row["baseline_losses"]) / len(row["baseline_losses"]),
                            "probe/loss_mean": sum(row["losses"]) / len(row["losses"]),
                            "probe/delta_mean": row["delta_mean"],
                            "probe/delta_sem": row["delta_sem"],
                        },
                        step=step,
                    )
        run.summary["probe/success"] = 1
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        tracker.finish()
        print(json.dumps(dict(identity=identity, output=str(destination), mode=args.mode)), flush=True)


if __name__ == "__main__":
    main()
