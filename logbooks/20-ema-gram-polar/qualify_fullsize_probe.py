# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify full-size probe execution on a pilot; do not produce confirmation evidence."""

import argparse
import hashlib
import json
import math
import os
import uuid
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import haliax as hax
from levanter.data.loader import DataLoader
from levanter.tracker.wandb import WandbConfig
from levanter.utils.tree_utils import inference_mode
from qualify_checkpoint_restore import resolve, restore_state, template_for

from experiments.ema_gram.probe import decomposition_diagnostics, loss_probes


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_probe(request, cfg, settings, report, output, tracker):
    state, restored = restore_state(request, cfg)
    report["restore"] = restored
    report["phase"] = "restored"
    write_report(output, report)
    tracker.log({"qualification/restored_counter": restored["restored_counter"]}, step=0)
    with cfg.trainer.use_device_mesh():
        dataset = cfg.data.validation_sets(cfg.model.max_Pos)[settings["dataset"]]
        loader = DataLoader(dataset, batch_size=cfg.trainer.EvalBatch, axis_resources=cfg.trainer.compute_axis_mapping)
        offset, count = settings["batch_offset"], settings["batches"]
        batches = list(islice(loader, offset, offset + count))
        if len(batches) != count:
            raise ValueError("Insufficient held-out qualification batches")
        report["eval_batch_size"] = cfg.trainer.eval_batch_size
        report["phase"] = "batches_loaded"
        write_report(output, report)

        @hax.named_jit(axis_resources=cfg.trainer.compute_axis_mapping)
        def loss_fn(model, batch):
            model = cfg.trainer.mp.cast_to_compute(inference_mode(model, True))
            return model.compute_next_token_loss(batch, key=None).scalar()

        losses = [
            asdict(row)
            for row in loss_probes(
                state, loss_fn, batches, settings["step_sizes"], polar_steps=cfg.optimizer.polar_steps
            )
        ]
        if len(losses) != 27:
            raise ValueError("Expected all nine alpha/ridge settings and three common step sizes")
        for row in losses:
            values = row["baseline_losses"] + row["losses"] + [row[k] for k in ("delta_mean", "delta_std", "delta_sem")]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Nonfinite full-size loss probe")
        report.update(loss_probes=losses, phase="loss_probes_complete")
        write_report(output, report)
        tracker.log({"qualification/finite_loss_probe_rows": len(losses)}, step=1)
        decompositions = [
            asdict(row)
            for row in decomposition_diagnostics(state.model, state.opt_state, polar_steps=cfg.optimizer.polar_steps)
        ]
        if len(decompositions) != 18 or {row["category"] for row in decompositions} != {"attention", "mlp"}:
            raise ValueError("Expected attention and MLP comparisons for all nine settings")
        for row in decompositions:
            values = [row["cosine"], row["relative_error"], *row["seconds"].values()]
            if any(value is None or not math.isfinite(value) for value in values):
                raise ValueError("Degenerate or nonfinite full-size decomposition comparison")
        report.update(decompositions=decompositions, phase="probe_complete")
        write_report(output, report)
        tracker.log({"qualification/finite_decomposition_rows": len(decompositions)}, step=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--mode", choices=("validate", "logging-check", "template-check", "probe"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.request.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.request_sha256:
        raise ValueError("Probe qualification request digest differs")
    settings = json.loads(raw)
    request, cfg = resolve(args.request.parent / settings["restore_request"], settings["restore_request_sha256"])
    if (settings["dataset"], settings["batch_offset"], settings["batches"], settings["step_sizes"]) != (
        "uncheatable_eval/wikipedia_english",
        16,
        2,
        [1e-5, 3e-5, 1e-4],
    ):
        raise ValueError("Unexpected frozen qualification probe settings")
    report = dict(
        mode=args.mode,
        source_commit=os.environ.get("GIT_COMMIT"),
        request_sha256=args.request_sha256,
        entrypoint_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        restore_entrypoint_sha256=hashlib.sha256(
            (args.request.parent / "qualify_checkpoint_restore.py").read_bytes()
        ).hexdigest(),
        task=os.environ.get("IRIS_TASK_ID"),
        attempt_uid=os.environ.get("IRIS_ATTEMPT_UID"),
        settings=settings,
        checkpoint=request["checkpoint"],
        scope="Pilot infrastructure qualification only; no confirmation evidence, optimizer update or checkpoint write.",
        phase="validated",
    )
    if args.mode == "validate":
        write_report(args.output, report)
        return
    if not report["source_commit"]:
        raise ValueError("Require a published source identity before ONLINE tracking")
    identity = "ema-gram-fullsize-probe-qualification-" + uuid.uuid4().hex[:12]
    tracker = WandbConfig(
        entity="marin-community",
        project="marin",
        id=identity,
        name=identity,
        group="ema-gram-20-fullsize-probe-qualification",
        mode="online",
        resume="never",
        background=False,
        save_code=False,
    ).init(None)
    print("W&B: https://wandb.ai/marin-community/marin/runs/" + identity, flush=True)
    report["identity"] = identity
    try:
        tracker.log_hyperparameters(dict(**report, restore_request=request))
        write_report(args.output, report)
        if args.mode == "template-check":
            template_for(request, cfg)
            report["full_state_template_matches_manifest"] = True
        if args.mode == "probe":
            run_probe(request, cfg, settings, report, args.output, tracker)
        report["success"] = True
        tracker.log({"qualification/success": 1}, step=3)
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        write_report(args.output, report)
        tracker.finish()


if __name__ == "__main__":
    main()
