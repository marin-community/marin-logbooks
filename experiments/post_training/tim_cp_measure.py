# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Session-only packed CP qualification; never part of the product PRs."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
from dataclasses import replace

import ray
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from omegaconf import OmegaConf
from ray._raylet import PlacementGroupID
from ray.util.placement_group import PlacementGroup, placement_group_table, remove_placement_group
from rigging.filesystem.storage_path import StoragePath
from skyrl_train.config.objective_spec import score_centering_tis_cap
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.distributed.megatron.megatron_strategy import MegatronStrategy
from skyrl_train.distributed.megatron.megatron_utils import preprocess_packed_seqs
from skyrl_train.entrypoints.main_base import BasePPOExp, run_ray_driver
from skyrl_train.objective.score_centering import ppo_tis_score_centering_correction
from skyrl_train.objective.score_centering_probe import hypothetical_score_probe
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import utils as runtime_utils
from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronForwardMicroBatch, MegatronPolicyMicroBatch
from skyrl_train.workers.megatron.megatron_worker import CriticWorker, MegatronPolicyWorkerBase, RefWorker

ARMS = ("tis", "sc", "probe")


def write_record(root, name, value):
    destination = StoragePath(root) / name
    with destination.open("w") as out:
        json.dump(value, out, indent=2, default=lambda x: x.item())
    print("TIM_CP_RECORD", str(destination), flush=True)


def route_fixture(actions, layers, top_k, experts, device):
    if not layers:
        return None
    # Balanced, distinct choices form a controlled routing replay. They are held
    # fixed for the reference, old-policy and every training comparison.
    positions = torch.arange(actions, device=device)[:, None, None]
    layer = torch.arange(layers, device=device)[None, :, None]
    slot = torch.arange(top_k, device=device)[None, None, :]
    return ((positions + 7 * layer + slot) % experts).to(torch.int16).unsqueeze(0)


def numeric_reference(wrapper):
    """Compare the real GPU scorer with dense values and gradients on the live mesh."""
    device = torch.cuda.current_device()
    reports = []
    dp_group = mpu.get_data_parallel_group(with_context_parallel=False)
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(31)
        parameters = torch.randn(7, 7, device=device, requires_grad=True)
        reference = parameters.detach().clone().requires_grad_()
        seq = torch.arange(3 * 27, device=device).reshape(3, 27) % 7
        attention = torch.ones_like(seq, dtype=torch.bool)
        attention[0, 4] = False
        attention[1, :3], attention[1, -2:] = False, False
        attention[2, :8], attention[2, -5:] = False, False
        mask = attention[:, -12:].float()
        mask[0, 4], mask[2, 1] = 0, 0
        # Different DP rows have different eligible-token counts.
        mask[0, : mpu.get_data_parallel_rank() % 4] = 0
        adv = torch.tensor([1.7, -2.3, 0.0] * 4, device=device).expand(3, -1).clone()
        adv[~mask.bool()] = torch.nan
        ids = torch.arange(3, device=device).expand(3, 12, 3).clone()
        ids[~mask.bool()] = -1
        o = torch.tensor([0.23, 0.21, 0.17, 0.13, 0.11, 0.09, 0.06], device=device)
        q = torch.tensor([0.31, 0.20, 0.13, 0.11, 0.10, 0.08, 0.07], device=device)
        packed_tokens, packed = preprocess_packed_seqs(seq, attention)
        wrapper.train()
        logits = parameters[packed_tokens].to(dtype)
        got = wrapper._response_logprobs(logits, seq, attention, 12, packed, ids)
        dense = reference[seq[:, -13:-1]].to(dtype).float().log_softmax(-1)
        valid = mask.bool()
        chosen = dense.gather(-1, seq[:, -12:, None]).squeeze(-1)
        torch.testing.assert_close(got.scores[valid], chosen[valid], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(got.selected_logprobs[valid], dense[..., :3][valid], atol=1e-5, rtol=1e-5)
        p = dense.exp()
        tail = p[..., 3:].sum(-1).detach()
        old = torch.cat((o[:3].expand(3, 12, -1), p[..., 3:].detach() * (o[3:].sum() / tail)[..., None]), -1)
        behavior = torch.cat((q[:3].expand(3, 12, -1), p[..., 3:].detach() * (q[3:].sum() / tail)[..., None]), -1)
        ratio = p / old
        safe_adv = adv.nan_to_num()[..., None]
        loss = (
            behavior
            * (old / behavior).clamp(max=1.3)
            * -torch.minimum(ratio * safe_adv, ratio.clamp(0.8, 1.2) * safe_adv)
        ).sum(-1)
        coefficient = -torch.autograd.grad(loss.sum(), dense, retain_graph=True)[0]
        tail_coefficient = coefficient[..., 3] / p[..., 3].detach()
        expected = ((coefficient[..., :3] - tail_coefficient[..., None] * p[..., :3]).detach() * dense[..., :3]).sum(-1)
        correction = ppo_tis_score_centering_correction(
            got.selected_logprobs,
            o[:3].log().expand_as(ids),
            q[:3].log().expand_as(ids),
            adv,
            mask,
            tis_cap=1.3,
            eps_clip_low=0.2,
            eps_clip_high=0.2,
        )
        torch.testing.assert_close(correction[valid], expected[valid], atol=1e-5, rtol=1e-5)
        weights = mask * torch.tensor([1.0, 0.5, 1.0] * 4, device=device)
        denominator = weights.sum().detach()
        dist.all_reduce(denominator, group=dp_group)
        ((correction * weights).sum() / denominator).backward()
        ((expected * weights).sum() / denominator).backward()
        dist.all_reduce(parameters.grad, group=mpu.get_context_parallel_group())
        dist.all_reduce(parameters.grad, group=dp_group)
        dist.all_reduce(reference.grad, group=dp_group)
        tolerance = 0.005 if dtype == torch.bfloat16 else 1e-6
        torch.testing.assert_close(parameters.grad, reference.grad, atol=tolerance, rtol=tolerance)
        reports.append(
            {
                "dtype": str(dtype),
                "score_max_abs": float((got.scores[valid] - chosen[valid]).abs().max()),
                "head_max_abs": float((got.selected_logprobs[valid] - dense[..., :3][valid]).abs().max()),
                "gradient_max_abs": float((parameters.grad - reference.grad).abs().max()),
                "gradient_mean_abs": float((parameters.grad - reference.grad).abs().mean()),
                "global_weight_count": float(denominator),
            }
        )
    return reports


class QualificationPolicyWorker(MegatronPolicyWorkerBase):
    def qualification_layout(self):
        config = self.strategy.hf_config
        return {
            "layers": self.model.router_replay.num_moe_layers_total if self.model.router_replay is not None else 0,
            "top_k": getattr(config, "num_experts_per_tok", 1),
            "experts": getattr(config, "num_experts", getattr(config, "num_local_experts", 1)),
        }

    def _ppo_train_impl(self, data, timing):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = super()._ppo_train_impl(data, timing)
        torch.cuda.synchronize()
        if not hasattr(self, "qualification_updates"):
            self.qualification_updates = []
        self.qualification_updates.append(
            {
                "rank": self._rank,
                "seconds": time.perf_counter() - started,
                "peak_allocated": torch.cuda.max_memory_allocated(),
                "peak_reserved": torch.cuda.max_memory_reserved(),
                "status": result.metadata["train_status"],
            }
        )
        return result

    def qualification_report(self):
        return getattr(self, "qualification_updates", [])

    def qualification_benchmark(self, token_ids, actions, reference_scores, label, warmups, repeats):
        self.backload_to_gpu()
        device = torch.cuda.current_device()
        dp = mpu.get_data_parallel_rank()
        cp = mpu.get_context_parallel_rank()
        pp = mpu.get_pipeline_model_parallel_rank()
        last = mpu.is_pipeline_last_stage(ignore_virtual=True)
        config = self.strategy.hf_config
        layers = self.model.router_replay.num_moe_layers_total if self.model.router_replay is not None else 0
        routes = route_fixture(
            actions,
            layers,
            getattr(config, "num_experts_per_tok", 1),
            getattr(config, "num_experts", getattr(config, "num_local_experts", 1)),
            device,
        )
        tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        attention = torch.ones_like(tokens)
        positions = torch.arange(len(token_ids), device=device).unsqueeze(0)
        self.model.eval()
        base_micro = MegatronForwardMicroBatch(tokens, attention, positions, actions, rollout_routed_experts=routes)
        captured = {}
        original_response = self.model._response_logprobs

        def capture(logits, sequences, attention_mask, num_actions, packed, *args):
            result = original_response(logits, sequences, attention_mask, num_actions, packed, *args)
            query_map = torch.zeros_like(sequences)
            query_map[:, -num_actions - 1 : -1] = torch.arange(1, num_actions + 1, device=device)
            local, _ = preprocess_packed_seqs(query_map, attention_mask.bool())
            row, col = local.nonzero(as_tuple=True)
            query = local[row, col] - 1
            values, indices = logits[..., : config.vocab_size].topk(128, dim=-1)
            selected = (
                result.scores[0, query, None]
                + values[row, col].float()
                - logits[row, col, sequences[0, -num_actions:][query]][:, None].float()
            )
            ids = torch.zeros((1, num_actions, 128), dtype=torch.long, device=device)
            scores = torch.zeros((1, num_actions, 128), device=device)
            ids[0, query], scores[0, query] = indices[row, col], selected
            dist.all_reduce(ids, group=mpu.get_context_parallel_group())
            dist.all_reduce(scores, group=mpu.get_context_parallel_group())
            captured.update(ids=ids, head=scores, sampled=result.scores.float())
            return result

        self.model._response_logprobs = capture
        try:
            with torch.no_grad():
                self.model.forward([base_micro], len(token_ids), 1, temperature=self.scoring_temperature)
        finally:
            self.model._response_logprobs = original_response
        if not last:
            captured = {
                "ids": torch.empty((1, actions, 128), dtype=torch.long, device=device),
                "head": torch.empty((1, actions, 128), device=device),
                "sampled": torch.empty((1, actions), device=device),
            }
        for value in captured.values():
            dist.broadcast(
                value, src=mpu.get_pipeline_model_parallel_last_rank(), group=mpu.get_pipeline_model_parallel_group()
            )
        captured["ids"] = captured["ids"].cpu()
        captured["head"] = captured["head"].cpu()
        del value
        mask = torch.ones((1, actions), device=device)
        mask[:, 7::97] = 0
        advantages = torch.tensor([1.7, -2.3, 0.0], device=device).repeat((actions + 2) // 3)[:actions].unsqueeze(0)
        probe_positions = torch.linspace(0, actions - 1, min(32, actions), device=device).round().long().unsqueeze(0)
        valid_probe = mask.gather(1, probe_positions).bool()
        q_head, q_sampled = captured["head"], captured["sampled"]
        base = MegatronPolicyMicroBatch(
            tokens,
            attention,
            positions,
            actions,
            q_sampled,
            reference_scores[dp : dp + 1].to(device),
            advantages,
            mask,
            q_sampled,
            None,
            correction_weights=torch.ones_like(mask),
            rollout_routed_experts=routes,
        )
        local_query_map = torch.zeros_like(tokens)
        local_query_map[:, -actions - 1 : -1] = torch.arange(1, actions + 1, device=device)
        local_queries, _ = preprocess_packed_seqs(local_query_map, attention.bool())
        owned = local_queries[local_queries > 0] - 1
        ownership = {
            "full": int(mask[0, owned].bool().sum()),
            "compact": int(torch.isin(owned, probe_positions[valid_probe]).sum()),
        }
        rng = MegatronStrategy.get_rng_state()
        tracker_state = copy.deepcopy(get_cuda_rng_tracker().get_states())
        rows = []
        numerical = numeric_reference(self.model)
        settings = self.cfg.trainer.algorithm
        initial_sc = settings.score_centering_enabled
        try:
            for repeat in range(-warmups, repeats):
                for arm in ARMS[repeat % 3 :] + ARMS[: repeat % 3]:
                    MegatronStrategy.load_rng_state(rng)
                    get_cuda_rng_tracker().set_states(copy.deepcopy(tracker_state))
                    settings.score_centering_enabled = arm == "sc"
                    input_transfer_started = time.perf_counter()
                    ids = (
                        captured["ids"].to(device)
                        if arm == "sc"
                        else (
                            captured["ids"].gather(1, probe_positions.cpu()[..., None].expand(-1, -1, 128)).to(device)
                            if arm == "probe"
                            else None
                        )
                    )
                    behavior_head = q_head.to(device) if arm == "sc" else None
                    torch.cuda.synchronize()
                    input_transfer_seconds = time.perf_counter() - input_transfer_started
                    if ids is not None:
                        eligibility = mask.bool() if arm == "sc" else valid_probe
                        ids[~eligibility] = -1
                    forward = replace(
                        base_micro,
                        score_topk_indices=ids,
                        score_probe_positions=probe_positions if arm == "probe" else None,
                    )
                    self.optimizer.zero_grad()
                    for chunk in self.actor_module:
                        chunk.zero_grad_buffer()
                    dist.barrier()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    self.model.eval()
                    with torch.no_grad():
                        old = self.model.forward([forward], len(token_ids), 1, temperature=self.scoring_temperature)
                    torch.cuda.synchronize()
                    forward_seconds = time.perf_counter() - started
                    transfer_started = time.perf_counter()
                    old_scores = old.scores.float() if last else torch.empty_like(q_sampled)
                    old_head = (
                        old.selected_logprobs
                        if last and ids is not None
                        else torch.empty_like(ids, dtype=torch.float32) if ids is not None else None
                    )
                    dist.broadcast(
                        old_scores,
                        src=mpu.get_pipeline_model_parallel_last_rank(),
                        group=mpu.get_pipeline_model_parallel_group(),
                    )
                    if old_head is not None:
                        dist.broadcast(
                            old_head,
                            src=mpu.get_pipeline_model_parallel_last_rank(),
                            group=mpu.get_pipeline_model_parallel_group(),
                        )
                    host_scores = old_scores.cpu()
                    host_head = old_head.cpu() if old_head is not None else None
                    old_scores = host_scores.to(device)
                    old_head = host_head.to(device) if host_head is not None else None
                    torch.cuda.synchronize()
                    transfer_seconds = time.perf_counter() - transfer_started
                    summary_seconds, probe_count = 0.0, 0
                    if arm == "probe" and last and cp == 0:
                        summary_started = time.perf_counter()
                        metrics = hypothetical_score_probe(
                            host_head,
                            q_head.gather(1, probe_positions.cpu()[..., None].expand(-1, -1, 128)),
                            host_scores.gather(1, probe_positions.cpu()),
                            q_sampled.gather(1, probe_positions).cpu(),
                            advantages.gather(1, probe_positions).cpu(),
                            valid_probe.cpu(),
                            torch.zeros_like(probe_positions.cpu()),
                            response_positions=probe_positions.cpu(),
                            response_lengths=torch.tensor([actions]),
                            training_step=0,
                            loss_tokens=int(mask.sum()),
                            tis_cap=score_centering_tis_cap(settings),
                            eps_clip_low=settings.eps_clip_low,
                            eps_clip_high=settings.eps_clip_high,
                        )
                        summary_seconds = time.perf_counter() - summary_started
                        probe_count = metrics["score_centering_probe/sampled_tokens"]
                    micro = replace(
                        base,
                        old_action_log_probs=old_scores,
                        score_topk_indices=ids if arm == "sc" else None,
                        score_old_logprobs=old_head if arm == "sc" else None,
                        score_behavior_logprobs=behavior_head,
                    )
                    self.model.train()
                    backward_started = time.perf_counter()
                    self.model.forward_backward_mini_batch(
                        [micro], len(token_ids), 1, temperature=self.scoring_temperature
                    )
                    torch.cuda.synchronize()
                    backward_seconds = time.perf_counter() - backward_started
                    elapsed = time.perf_counter() - started
                    fingerprints = torch.zeros(2, dtype=torch.float64, device=device)
                    for chunk in self.actor_module:
                        for parameter in chunk.parameters():
                            grad = getattr(parameter, "main_grad", parameter.grad)
                            if grad is not None:
                                # Bounded reductions: no whole-model FP32 copy.
                                for part in grad.flatten().split(262144):
                                    fingerprints[0] += part.double().sum()
                                    fingerprints[1] += part.double().square().sum()
                    rows.append(
                        {
                            "rank": self._rank,
                            "dp": dp,
                            "cp": cp,
                            "pp": pp,
                            "label": label,
                            "arm": arm,
                            "repeat": repeat,
                            "seconds": elapsed,
                            "forward_seconds": forward_seconds,
                            "backward_seconds": backward_seconds,
                            "host_transfer_seconds": transfer_seconds,
                            "input_evidence_transfer_seconds": input_transfer_seconds,
                            "summary_seconds": summary_seconds,
                            "probe_count": probe_count,
                            "peak_allocated": torch.cuda.max_memory_allocated(),
                            "peak_reserved": torch.cuda.max_memory_reserved(),
                            "gradient_sum": float(fingerprints[0]),
                            "gradient_l2_squared": float(fingerprints[1]),
                            "old_head_bytes": (
                                host_head.numel() * host_head.element_size() if host_head is not None else 0
                            ),
                            "owned_queries": (
                                ownership["full" if arm == "sc" else "compact"] if arm != "tis" and last else 0
                            ),
                            "nonpadding_tokens": len(token_ids),
                            "response_tokens": actions,
                        }
                    )
                    print(
                        "TIM_CP_BENCH",
                        json.dumps(
                            {k: v for k, v in rows[-1].items() if k not in {"gradient_sum", "gradient_l2_squared"}}
                        ),
                        flush=True,
                    )
                    del micro, forward, old, old_head, host_head, old_scores, ids, behavior_head
        finally:
            settings.score_centering_enabled = initial_sc
            self.optimizer.zero_grad()
            for chunk in self.actor_module:
                chunk.zero_grad_buffer()
        return {
            "rows": rows,
            "numerical": numerical,
            "rank": self._rank,
            "geometry": {
                "cp": mpu.get_context_parallel_world_size(),
                "dp": dist.get_world_size(mpu.get_data_parallel_group(with_context_parallel=False)),
                "pp": mpu.get_pipeline_model_parallel_world_size(),
                "ep": mpu.get_expert_model_parallel_world_size(),
                "tp": mpu.get_tensor_model_parallel_world_size(),
            },
            "hardware": torch.cuda.get_device_name(),
            "model": config.to_dict(),
            "tokens_sha256": hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),
        }


PolicyWorker = ray.remote(num_gpus=1)(QualificationPolicyWorker)


def reference_forward(trainer, token_ids, actions):
    dp = trainer.policy_model.actor_infos[0].rank.dp_size
    tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).expand(dp, -1).clone()
    if trainer.ref_model is None:
        return torch.zeros((dp, actions))
    layout = ray.get(trainer.policy_model.async_run_ray_method("pass_through", "qualification_layout"))[0]
    routes = route_fixture(actions, layout["layers"], layout["top_k"], layout["experts"], "cpu")
    fields = {"sequences": tokens, "attention_mask": torch.ones_like(tokens)}
    if routes is not None:
        fields["rollout_routed_experts"] = routes.expand(dp, -1, -1, -1).clone()
    batch = TrainingInputBatch(fields)
    batch.metadata = {"response_length": actions, "global_step": trainer.global_step}
    trainer.policy_model.offload_to_cpu()
    ray.get(trainer.policy_model.async_run_ray_method("pass_through", "empty_cache"))
    trainer.ref_model.backload_to_gpu()
    try:
        outputs = ray.get(trainer.ref_model.async_run_ray_method("mesh", "forward", batch))
        return concatenate_outputs_after_mesh_dispatch(trainer.ref_model.actor_infos, outputs)["output"]
    finally:
        trainer.ref_model.offload_to_cpu()
        ray.get(trainer.ref_model.async_run_ray_method("pass_through", "empty_cache"))


@ray.remote(num_cpus=1, max_retries=0, max_calls=1)
def run_arm(cfg, arm, smoke):
    cfg = copy.deepcopy(cfg)
    cfg.trainer.run_name += "-" + arm
    cfg.trainer.algorithm.score_centering_enabled = arm == "sc"
    cfg.trainer.algorithm.score_centering_probe.interval = 1 if arm == "probe" else 0
    before_groups = set(placement_group_table())
    exp = BasePPOExp(cfg)
    exp.get_worker_classes = lambda: (PolicyWorker, CriticWorker, RefWorker)
    root = cfg.trainer.ckpt_path.rstrip("/") + "/qualification/" + arm
    with exp.get_tracker() as tracker:
        trainer = None
        try:
            trainer = exp._setup_trainer(tracker)
            records, consumed = [], []
            original_log = trainer._log_metrics_stdout
            original_finish = trainer._finish_score_probe

            def log(payload, step, kind="train"):
                records.append({"step": step, "kind": kind, "metrics": dict(payload)})
                return original_log(payload, step, kind)

            def finish(batch):
                original_finish(batch)
                assert not any(str(key).startswith("score_") for key in batch.keys())
                consumed.append(
                    {
                        "step": trainer.global_step,
                        "sampled_tokens": trainer.all_metrics["score_centering_probe/sampled_tokens"],
                    }
                )

            trainer._log_metrics_stdout, trainer._finish_score_probe = log, finish

            def collect():
                updates = ray.get(trainer.policy_model.async_run_ray_method("pass_through", "qualification_report"))
                write_record(
                    root,
                    "live.json",
                    {
                        "arm": arm,
                        "records": records,
                        "updates": updates,
                        "consumed": consumed,
                        "config": OmegaConf.to_container(cfg, resolve=True),
                    },
                )
                assert all(any(row["status"].get("raw_grad_norm", 0) > 0 for row in rank_rows) for rank_rows in updates)
                if arm == "probe":
                    assert consumed and sum(row["sampled_tokens"] for row in consumed) > 0
                if arm == "sc":
                    token_seed = trainer.tokenizer.encode(
                        "A careful benchmark compares the same tokens and model weights. ", add_special_tokens=False
                    )
                    sizes = [("smoke", 128, 64)] if smoke else [("short", 2048, 1024), ("long", 102400, 49152)]
                    for label, length, actions in sizes:
                        token_ids = (token_seed * ((length + len(token_seed) - 1) // len(token_seed)))[:length]
                        reference = reference_forward(trainer, token_ids, actions)
                        trainer.policy_model.backload_to_gpu()
                        results = ray.get(
                            trainer.policy_model.async_run_ray_method(
                                "pass_through",
                                "qualification_benchmark",
                                token_ids,
                                actions,
                                reference,
                                label,
                                1,
                                2 if smoke else 5,
                            )
                        )
                        write_record(root, f"{label}-benchmark.json", results)
                return {"arm": arm, "records_uri": root + "/live.json", "updates": len(updates), "consumed": consumed}

            original_finalize = trainer._finalize_training
            result = []

            async def finalize(*args, **kwargs):
                await original_finalize(*args, **kwargs)
                await trainer.context.close()
                result.append(collect())

            trainer._finalize_training = finalize
            asyncio.run(trainer.train())
            assert len(result) == 1
            return result[0]
        finally:
            if trainer is not None:
                asyncio.run(trainer.shutdown())
            for group_id in set(placement_group_table()) - before_groups:
                remove_placement_group(PlacementGroup(PlacementGroupID.from_hex(group_id)))


@ray.remote(num_cpus=1, max_retries=0)
def qualification_entrypoint(cfg):
    smoke = cfg.trainer.placement.policy_num_nodes == 1
    results = []
    for arm in (("sc",) if smoke else ("sc", "probe", "tis")):
        results.append(ray.get(run_arm.remote(cfg, arm, smoke)))
    write_record(cfg.trainer.ckpt_path.rstrip("/") + "/qualification", "completed.json", results)


def run(cfg):
    if cfg.trainer.placement.policy_num_nodes == 1:
        prepare_environment = runtime_utils.prepare_runtime_environment

        def smoke_environment(config):
            return {**prepare_environment(config), "NCCL_NVLS_ENABLE": "0"}

        runtime_utils.prepare_runtime_environment = smoke_environment
    run_ray_driver(cfg, qualification_entrypoint)
