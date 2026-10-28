# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record bounded evidence around the unmodified asynchronous training loop."""

import asyncio
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from skyrl_train.entrypoints.main_base import BasePPOExp, run_ray_driver
from torch.utils.data import Subset
from torchdata.stateful_dataloader import StatefulDataLoader

from experiments.post_training.sc_default_data import write_json


def install_measurements(trainer, entrypoint_started: float) -> None:
    cfg = trainer.cfg
    smoke = cfg.trainer.placement.colocate_all
    expected_rows = 10 if smoke else 10_000
    if len(trainer.eval_dataset) != expected_rows:
        raise ValueError(f"Evaluation rows were filtered: {len(trainer.eval_dataset)} != {expected_rows}")
    if cfg.trainer.algorithm.score_centering_probe.interval:
        raise ValueError("Hypothetical probes must remain disabled")
    root = str(cfg.trainer.export_path).rstrip("/") + "/study"
    local = Path("/tmp/sc-default-study-curves.jsonl")
    completed, excluded, syncs = [], [], []
    started = None
    original_publish = trainer._publish
    original_step = trainer._train_step
    original_evaluation = trainer._evaluate_and_log
    original_log = trainer._log_metrics_stdout
    original_sync = trainer._sync_policy_for_rollouts
    original_training = trainer._run_training
    engine_actors = [engine.inference_engine_actor for engine in trainer.inference_engine_client.engines]
    engine_info = ray.get([actor.performance_info.remote() for actor in engine_actors])
    if not all(
        "39e62869693c" in row["vllm_version"] and row["use_v2_model_runner"] and not row["logits_processors"]
        for row in engine_info
    ):
        raise ValueError("The qualified pinned vLLM V2 path changed")
    write_json(root + "/engine-runtime.json", engine_info)
    full_loader = trainer.eval_dataloader
    curve_loader = StatefulDataLoader(
        trainer.eval_dataset if smoke else Subset(trainer.eval_dataset, range(1000)),
        batch_size=cfg.trainer.eval_batch_size,
        shuffle=False,
        collate_fn=trainer.eval_dataset.collate_fn,
        num_workers=0,
        drop_last=False,
    )

    async def publish(step):
        nonlocal started
        if started is None:
            started = time.perf_counter()
            write_json(root + "/startup.json", {"entrypoint_to_first_publish_seconds": started - entrypoint_started})
        await original_publish(step)

    async def step(epoch, step_wall):
        batch = await original_step(epoch, step_wall)
        now = time.perf_counter()
        if started is None:
            raise RuntimeError("Training began without the first rollout publication")
        paused = sum(end - begin for begin, end in excluded if begin >= started and end <= now)
        completed.append(
            {
                "step": trainer.global_step,
                "elapsed_including_evaluation_seconds": now - started,
                "training_cycle_elapsed_seconds": now - started - paused,
                "excluded_evaluation_seconds": paused,
                "trained_loss_tokens": int(batch["loss_mask"].sum()),
                "response_tokens": int(batch["response_mask"].sum()),
                "response_lengths": batch["response_mask"].sum(-1).tolist(),
            }
        )
        trainer.all_metrics["study/training_cycle_elapsed_seconds"] = now - started - paused
        write_json(root + f"/cycle-{trainer.global_step:04d}.json", completed[-1])
        return batch

    async def evaluation(state, *, commit):
        # The shared 1,000-task curve panel is a balanced prefix of the fixed
        # 10,000-task final panel. Endpoint selection never uses observed scores.
        trainer.eval_dataloader = full_loader if state.global_step == cfg.trainer.max_steps else curve_loader
        begin = time.perf_counter()
        try:
            await original_evaluation(state, commit=commit)
        finally:
            excluded.append((begin, time.perf_counter()))
            trainer.eval_dataloader = full_loader
        write_json(
            root + f"/evaluation-time-{state.global_step:04d}.json",
            {
                "step": state.global_step,
                "seconds": excluded[-1][1] - begin,
                "expected_tasks": expected_rows if smoke or state.global_step == cfg.trainer.max_steps else 1000,
            },
        )

    async def training(batch, *, step_wall=None):
        if cfg.trainer.algorithm.score_centering_enabled:
            scores = batch["score_behavior_logprobs"]
            valid = batch["loss_mask"].bool().flatten().nonzero().flatten()
            # Summarize already captured q on a bounded, evenly spaced token
            # sample. This adds no model requests or gradient passes.
            positions = valid[torch.linspace(0, len(valid) - 1, min(len(valid), 4096)).long()]
            mass = scores.flatten(0, 1)[positions].float().exp().sum(-1).numpy()
            trainer.all_metrics.update(
                {
                    "study/captured_mass_sampled_tokens": len(mass),
                    "study/captured_mass_mean": float(mass.mean()),
                    "study/captured_mass_min": float(mass.min()),
                    "study/captured_mass_p10": float(np.quantile(mass, 0.1)),
                    "study/captured_mass_median": float(np.median(mass)),
                    "study/candidate_array_bytes": (
                        scores.numel() * scores.element_size()
                        + batch["score_topk_indices"].numel() * batch["score_topk_indices"].element_size()
                    ),
                }
            )
            if scores.shape[-1] != cfg.trainer.algorithm.score_centering_top_k:
                raise ValueError("Candidate width differs from the configured arm")
        elif "score_behavior_logprobs" in batch or "score_topk_indices" in batch:
            raise ValueError("Baseline unexpectedly captured candidates")
        result = await original_training(batch, step_wall=step_wall)
        if smoke and batch.get("base_action_log_probs") is None:
            raise ValueError("Qwen qualification did not score the reference model")
        return result

    async def sync(*, reason):
        await original_sync(reason=reason)
        if not smoke:
            return
        name = "model.layers.0.self_attn.o_proj.weight"
        ranks = await asyncio.gather(*(actor.read_engine_weights.remote([name], False) for actor in engine_actors))
        hashes = []
        for engine in ranks:
            for row in engine:
                value = row[name]
                if not value["found"]:
                    raise ValueError(f"Serving weight is missing after wake-up: {value}")
                tensor = value["tensor"]
                if not torch.isfinite(tensor).all() or not tensor.abs().sum() > 0:
                    raise ValueError("Serving weights were not restored after sleep")
                hashes.append(hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
        if len(set(hashes)) != 1:
            raise ValueError("Inference replicas disagree after weight publication")
        syncs.append(
            {"step": trainer.global_step, "reason": reason, "weight_sha256": hashes[0], "replicas": len(hashes)}
        )
        write_json(root + f"/weights-{len(syncs):04d}.json", syncs[-1])

    def log(payload, step, kind="train"):
        original_log(payload, step=step, kind=kind)
        record = {"step": step, "kind": kind, "monotonic_seconds": time.perf_counter(), "metrics": payload}
        with local.open("a") as stream:
            stream.write(json.dumps(record, default=float) + "\n")
        write_json(root + f"/metrics-{kind}-{step:04d}.json", record)
        if kind == "eval":
            expected = expected_rows if smoke or step == cfg.trainer.max_steps else 1000
            if payload.get("eval/all/num_scored") != expected:
                raise RuntimeError(
                    "Evaluation has missing or infrastructure-failed grades; preserve answers for regrading"
                )
        if kind == "train" and step == cfg.trainer.max_steps:
            if smoke and len({row["weight_sha256"] for row in syncs}) < 2:
                raise ValueError("Qualification did not change published inference weights")
            write_json(
                root + "/completed-training.json",
                {
                    "steps": completed,
                    "weight_publications": syncs,
                    "main_reference_scoring_enabled": bool(cfg.trainer.algorithm.use_kl_loss),
                    "config_sha256": hashlib.sha256(OmegaConf.to_yaml(cfg, resolve=True).encode()).hexdigest(),
                    "limits": (
                        "Cycle span excludes measured evaluation sessions. Phase timers overlap and must not be summed. "
                        "Startup, checkpoint/export, and terminal teardown are separate."
                    ),
                },
            )

    trainer._publish = publish
    trainer._train_step = step
    trainer._evaluate_and_log = evaluation
    trainer._log_metrics_stdout = log
    trainer._sync_policy_for_rollouts = sync
    trainer._run_training = training


@ray.remote(num_cpus=1, max_retries=0)
def study_entrypoint(cfg) -> None:
    started = time.perf_counter()
    experiment = BasePPOExp(cfg)
    setup = experiment._setup_trainer

    def measured_setup(tracker):
        trainer = setup(tracker)
        install_measurements(trainer, started)
        return trainer

    experiment._setup_trainer = measured_setup
    experiment.run()


def run(cfg) -> None:
    run_ray_driver(cfg, study_entrypoint)
