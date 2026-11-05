# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Observe complete cycles and consumed probes without serializing sampler and trainer."""

import asyncio
import hashlib
import json
import time

import ray
import torch
from omegaconf import OmegaConf
from rigging.filesystem.buckets import filesystem_for
from skyrl_train.entrypoints.main_base import BasePPOExp, run_ray_driver


def write_json(uri: str, value) -> None:
    filesystem, path = filesystem_for(uri)
    with filesystem.open(path, "wt") as destination:
        json.dump(value, destination, indent=2, default=float)


def install_measurements(trainer, entrypoint_started: float) -> None:
    cfg = trainer.cfg
    if cfg.trainer.algorithm.score_centering_enabled:
        raise ValueError("This comparison requires ordinary PPO/TIS with SC off")
    root = str(cfg.trainer.export_path).rstrip("/") + "/probe-cost"
    completed, probes, publications = [], [], []
    started = None
    original_publish = trainer._publish
    original_step = trainer._train_step
    original_log = trainer._log_metrics_stdout
    original_finish = trainer._finish_score_probe
    original_optimize = trainer.train_critic_and_policy
    original_sync = trainer._sync_policy_for_rollouts
    smoke = cfg.trainer.placement.colocate_all
    engine_actors = [engine.inference_engine_actor for engine in trainer.inference_engine_client.engines]
    engine_info = ray.get([actor.performance_info.remote() for actor in engine_actors])
    if not all(
        "39e62869693c" in row["vllm_version"] and row["use_v2_model_runner"] and not row["logits_processors"]
        for row in engine_info
    ):
        raise ValueError("The qualified pinned vLLM V2 path changed")
    write_json(root + "/engine-runtime.json", engine_info)
    write_json(root + "/config.json", OmegaConf.to_container(cfg, resolve=True))

    async def publish(step):
        nonlocal started
        if started is None:
            started = time.perf_counter()
            write_json(root + "/startup.json", {"entrypoint_to_first_publish_seconds": started - entrypoint_started})
        return await original_publish(step)

    async def step(epoch, step_wall):
        batch = await original_step(epoch, step_wall)
        now = time.perf_counter()
        if started is None:
            raise RuntimeError("Training began without initial rollout publication")
        completed.append(
            {
                "step": trainer.global_step,
                "complete_cycle_elapsed_seconds": now - started,
                "trained_loss_tokens": int(batch["loss_mask"].sum()),
                "response_tokens": int(batch["response_mask"].sum()),
                "response_lengths": batch["response_mask"].sum(-1).tolist(),
                "metrics": dict(trainer.all_metrics),
                "timings": dict(trainer.all_timings),
            }
        )
        write_json(root + f"/cycle-{trainer.global_step:04d}.json", completed[-1])
        return batch

    def finish_probe(batch):
        positions = batch["score_probe_positions"]
        valid = (batch["score_topk_indices"] >= 0).all(-1) & batch["loss_mask"].gather(1, positions).bool()
        lengths = batch["response_mask"].sum(-1)
        probes.append(
            {
                "training_step": trainer.global_step,
                "positions": positions[valid].tolist(),
                "source_policy_steps": batch["score_probe_source_steps"][valid].tolist(),
                "response_lengths": lengths[valid.any(-1)].tolist(),
                "sampled_positions_per_response": valid.sum(-1)[valid.any(-1)].tolist(),
                "long_response_positions": int((valid & (lengths[:, None] >= 2048)).sum()),
            }
        )
        original_finish(batch)
        probes[-1]["metrics"] = {key: value for key, value in trainer.all_metrics.items() if key.startswith("score_")}

    def optimize(batch):
        if any(key.startswith("score_") for key in batch):
            raise ValueError("Probe evidence reached the optimizer")
        return original_optimize(batch)

    async def sync(*, reason):
        result = await original_sync(reason=reason)
        row = {"step": trainer.global_step, "reason": reason}
        if smoke:
            name = "model.layers.0.self_attn.o_proj.weight"
            ranks = await asyncio.gather(*(actor.read_engine_weights.remote([name], False) for actor in engine_actors))
            hashes = []
            for engine in ranks:
                for value in engine:
                    tensor = value[name]["tensor"]
                    if not torch.isfinite(tensor).all() or not tensor.abs().sum() > 0:
                        raise ValueError("Serving weights were not restored after publication")
                    hashes.append(hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
            if len(set(hashes)) != 1:
                raise ValueError("Inference replicas disagree after weight publication")
            row["weight_sha256"] = hashes[0]
        publications.append(row)
        return result

    def log(payload, step, kind="train"):
        original_log(payload, step=step, kind=kind)
        write_json(root + f"/metrics-{kind}-{step:04d}.json", {"step": step, "kind": kind, "metrics": payload})
        if kind == "train" and step == cfg.trainer.max_steps:
            if smoke and len({row["weight_sha256"] for row in publications}) < 2:
                raise ValueError("Qualification did not change published inference weights")
            write_json(
                root + "/completed.json",
                {"cycles": completed, "probes": probes, "publications": publications, "optimizer_isolation": True},
            )

    trainer._publish = publish
    trainer._train_step = step
    trainer._log_metrics_stdout = log
    trainer._finish_score_probe = finish_probe
    trainer.train_critic_and_policy = optimize
    trainer._sync_policy_for_rollouts = sync


@ray.remote(num_cpus=1, max_retries=0)
def measured_entrypoint(cfg) -> None:
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
    run_ray_driver(cfg, measured_entrypoint)
