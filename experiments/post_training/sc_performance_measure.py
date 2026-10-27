# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Interleave complete cycles and matched trainer replays on one allocation."""

import asyncio
import copy
import hashlib
import io
import math
import threading
import time

import httpx
import numpy as np
import pynvml
import ray
import torch
from omegaconf import OmegaConf
from ray._raylet import PlacementGroupID
from ray.util.placement_group import PlacementGroup, placement_group_table, remove_placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from rigging.filesystem.storage_path import StoragePath
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.inference_engines.chat_continuation import EXACT_PROMPT_TOKEN_IDS_KEY
from skyrl_train.inference_engines.response_topk import CHAT_TOPK_WIDTH_KEY, CHOICE_TOPK_KEY
from skyrl_train.rollouts.context import RolloutRequestSpec
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.model_clients import _render_chat_prompt

from experiments.post_training.tim_cp_measure import (
    CriticWorker,
    PolicyWorker,
    RefWorker,
    reference_forward,
    write_record,
)

ARMS = ("tis", "sc128", "sc32", "probe128", "probe32")


def schedule(repeats):
    return [
        (repeat, arm)
        for repeat in range(-1, repeats)
        for arm in ARMS[(repeat + 1) % len(ARMS) :] + ARMS[: (repeat + 1) % len(ARMS)]
    ]


def public_config(value):
    if isinstance(value, dict):
        return {
            key: (
                "<redacted>"
                if key.lower() in {"api_key", "password", "secret", "authorization"}
                else public_config(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [public_config(item) for item in value]
    return value


def batch_hash(batch):
    digest = hashlib.sha256()
    for name in ("sequences", "attention_mask", "response_mask", "loss_mask"):
        digest.update(batch[name].contiguous().numpy().tobytes())
    if batch.routed_expert_rows is not None:
        for row in batch.routed_expert_rows.rows:
            digest.update(row.tobytes())
    if batch.get("rollout_routed_experts") is not None:
        digest.update(batch["rollout_routed_experts"].contiguous().numpy().tobytes())
    return digest.hexdigest()


def copy_batch(base, keys):
    batch = TrainingInputBatch(
        {key: base[key].clone() if base[key] is not None else None for key in keys},
        routed_expert_rows=base.routed_expert_rows,
    )
    batch.metadata = copy.deepcopy(base.metadata)
    return batch


def write_arrays(root, name, arrays):
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    raw = output.getvalue()
    destination = StoragePath(root) / name
    with destination.open("wb") as target:
        target.write(raw)
    return {"uri": str(destination), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


@ray.remote(num_cpus=0)
class DeviceMemory:
    def __init__(self):
        pynvml.nvmlInit()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(pynvml.nvmlDeviceGetCount())]
        assert len(self.handles) == 8
        self.lock, self.stopped = threading.Lock(), threading.Event()
        self.label, self.peaks, self.samples = "startup", [0] * len(self.handles), 0
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def _sample(self):
        self.peaks = [
            max(peak, pynvml.nvmlDeviceGetMemoryInfo(handle).used)
            for peak, handle in zip(self.peaks, self.handles, strict=True)
        ]
        self.samples += 1

    def _poll(self):
        while not self.stopped.wait(0.05):
            with self.lock:
                self._sample()

    def reset(self, label):
        with self.lock:
            self.label, self.peaks, self.samples = label, [0] * len(self.handles), 0
            self._sample()

    def report(self, label):
        with self.lock:
            assert self.label == label
            self._sample()
            return {
                "node": ray.get_runtime_context().get_node_id(),
                "label": label,
                "used_bytes": list(self.peaks),
                "samples": self.samples,
                "sampling_interval": 0.05,
            }

    def close(self):
        self.stopped.set()
        self.thread.join()
        pynvml.nvmlShutdown()


async def choose_arm(trainer, arm):
    enabled = arm.startswith("sc")
    count = int(arm[2:] if enabled else arm[5:]) if arm != "tis" else 128
    trainer.cfg.trainer.algorithm.score_centering_enabled = enabled
    trainer.cfg.trainer.algorithm.score_centering_top_k = count
    trainer.cfg.trainer.algorithm.score_centering_probe.interval = 1 if arm.startswith("probe") else 0
    trainer.context.request_spec = RolloutRequestSpec.from_config(trainer.cfg)
    await asyncio.gather(
        *trainer.policy_model.async_run_ray_method("pass_through", "performance_set_arm", enabled, count)
    )
    return count


async def reset_memory(trainer, observers, label):
    await asyncio.gather(*(observer.reset.remote(label) for observer in observers))
    await asyncio.gather(*trainer.policy_model.async_run_ray_method("pass_through", "performance_reset_peaks"))


async def memory_report(trainer, observers, label):
    ranks = await asyncio.gather(*trainer.policy_model.async_run_ray_method("pass_through", "performance_peaks"))
    devices = await asyncio.gather(*(observer.report.remote(label) for observer in observers))
    return {"ranks": ranks, "devices": devices}


def replay_batch(base, arm):
    count = 32 if arm.endswith("32") else 128
    batch = copy_batch(base, [key for key in base if not key.startswith("score_")])
    if arm == "tis":
        return batch
    ids, behavior = base["score_topk_indices"][..., :count], base["score_behavior_logprobs"][..., :count]
    if arm.startswith("sc"):
        batch.update({"score_topk_indices": ids.clone(), "score_behavior_logprobs": behavior.clone()})
        return batch
    lengths = batch["response_mask"].sum(-1)
    width = min(32, int(lengths.max()))
    positions = torch.zeros((len(lengths), width), dtype=torch.long)
    valid = torch.zeros_like(positions, dtype=torch.bool)
    for row, length in enumerate(lengths.tolist()):
        size = min(width, length)
        positions[row, :size] = torch.linspace(0, length - 1, size).round().long()
        valid[row, :size] = True
    valid &= batch["loss_mask"].gather(1, positions).bool()
    gather = positions[..., None].expand(-1, -1, count)
    batch.update(
        {
            "score_topk_indices": ids.gather(1, gather).masked_fill(~valid[..., None], -1),
            "score_behavior_logprobs": behavior.gather(1, gather).masked_fill(~valid[..., None], torch.nan),
            "score_probe_positions": positions,
            "score_probe_source_steps": torch.full_like(positions, -1),
        }
    )
    return batch


async def matched_training(trainer, base, observers, root, smoke):
    records, expected = [], batch_hash(base)
    await asyncio.gather(*trainer.policy_model.async_run_ray_method("pass_through", "performance_rng"))
    for repeat, arm in schedule(2 if smoke else 5):
        await choose_arm(trainer, arm)
        await asyncio.gather(*trainer.policy_model.async_run_ray_method("pass_through", "performance_rng", True))
        label = f"matched-{repeat}-{arm}"
        await reset_memory(trainer, observers, label)
        trainer.all_metrics, trainer.all_timings = {}, {}
        started = time.perf_counter()
        batch = replay_batch(base, arm)
        prepared = time.perf_counter()
        await trainer._run_training(batch)
        elapsed = time.perf_counter() - started
        assert batch_hash(batch) == expected
        record = {
            "arm": arm,
            "repeat": repeat,
            "seconds": elapsed,
            "input_preparation_seconds": prepared - started,
            "batch_hash": expected,
            "loss_tokens": int(batch["loss_mask"].bool().sum()),
            "metrics": dict(trainer.all_metrics),
            "timings": dict(trainer.all_timings),
            "memory": await memory_report(trainer, observers, label),
        }
        if arm.startswith("probe"):
            assert record["metrics"]["score_centering_probe/sampled_tokens"] > 0
        records.append(record)
        write_record(root, "matched-training.json", records)
        print("SC_PERF_MATCHED", arm, repeat, elapsed, flush=True)
    return records


async def chat_payload(client, messages, width, *, seed=97, max_tokens=16):
    render = {"json": {"model": client.model_name, "messages": messages, "add_generation_prompt": True}, "headers": {}}
    rendered, prompt_ids = await _render_chat_prompt(client.tokenize, render, None)
    body = {
        "model": client.model_name,
        "messages": rendered,
        EXACT_PROMPT_TOKEN_IDS_KEY: prompt_ids,
        "logprobs": True,
        "top_logprobs": width + 1 if width else 0,
        "max_completion_tokens": max_tokens,
        "seed": seed,
        "return_tokens_as_token_ids": True,
        "include_stop_str_in_output": True,
    }
    payload = {"json": body, "headers": {"x-request-id": f"sc-perf-{seed}-{width}"}}
    if width:
        payload[CHAT_TOPK_WIDTH_KEY] = width
    return payload, prompt_ids


async def inference_comparison(trainer, root, smoke):
    client = trainer.inference_engine_client
    await client.wake_up()
    engine = client.engines[0].inference_engine_actor
    messages = [{"role": "user", "content": "Explain how to check an arithmetic result carefully."}]
    profiles = []
    for width in (0, 32, 128):
        payload, _ = await chat_payload(client, messages, width)
        profiles.append({"width": width, "profile": await engine.performance_profile.remote(payload)})
    traces = await engine.performance_export_profiles.remote(root)
    write_record(root, "inference-profiles.json", {"cpu": profiles, "traces": traces})
    endpoint = await engine.performance_start_route.remote()
    comparisons = []
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=300) as http:
            for repeat in range(-1, 2 if smoke else 5):
                for width in (0, 32, 128):
                    payload, prompt = await chat_payload(client, messages, width, seed=97 + max(repeat, 0))
                    request = {
                        "prompt_token_ids": prompt,
                        "candidate_count": width,
                        "sampling_params": {"seed": 97 + max(repeat, 0), "max_tokens": 16},
                    }
                    timings, results = {}, {}
                    routes = ("chat", "numeric") if repeat % 2 else ("numeric", "chat")
                    for route in routes:
                        started = time.perf_counter()
                        if route == "chat":
                            response = await engine.chat_completion.remote(copy.deepcopy(payload))
                            choice = response["choices"][0]
                            selected = choice.get(CHOICE_TOPK_KEY)
                            results[route] = {
                                "token_ids": np.asarray(choice["token_ids"]),
                                "head": selected,
                                "message_fields": list(choice["message"]),
                            }
                        else:
                            response = await http.post(endpoint, json=request)
                            response.raise_for_status()
                            with np.load(io.BytesIO(response.content), allow_pickle=False) as arrays:
                                results[route] = {key: arrays[key].copy() for key in arrays.files}
                            timings["numeric_response_bytes"] = len(response.content)
                        timings[route] = time.perf_counter() - started
                    same_tokens = np.array_equal(results["chat"]["token_ids"], results["numeric"]["token_ids"])
                    parity = {"same_token_ids": same_tokens}
                    if width and same_tokens:
                        parity["same_head_ids"] = bool(
                            np.array_equal(results["chat"]["head"].topk_indices, results["numeric"]["candidate_ids"])
                        )
                        parity["head_max_abs_error"] = float(
                            np.max(
                                np.abs(results["chat"]["head"].topk_logprobs - results["numeric"]["candidate_logprobs"])
                            )
                        )
                    comparisons.append(
                        {
                            "repeat": repeat,
                            "width": width,
                            "timings": timings,
                            "parity": parity,
                            "native_message_fields": results["chat"]["message_fields"],
                            "numeric_fields": list(results["numeric"]),
                        }
                    )
            # Use one transport for TIS, both heads and the actual-q collector.
            captured = []
            methods = (("tis", 0, False), ("head32", 32, False), ("head128", 128, False))
            if trainer.cfg.generator.engine_init_kwargs.get("logits_processors"):
                methods += (("iid8", 0, True),)
            for repeat in range(-1, 2 if smoke else 5):
                _, prompt = await chat_payload(client, messages, 0)
                outputs, block = {}, []
                offset = (repeat + 1) % len(methods)
                for method, width, iid in methods[offset:] + methods[:offset]:
                    started = time.perf_counter()
                    arrays = await engine.performance_generate.remote(
                        {
                            "prompt_token_ids": prompt,
                            "candidate_count": width,
                            "sampled8": iid,
                            "sampling_params": {"max_tokens": 16, "seed": 97},
                        }
                    )
                    elapsed = time.perf_counter() - started
                    outputs[method] = arrays["token_ids"]
                    block.append(
                        {
                            "method": method,
                            "repeat": repeat,
                            "seconds": elapsed,
                            "response_tokens": len(arrays["token_ids"]),
                            "candidate_payload_bytes": (
                                arrays["iid_indices"].nbytes + arrays["iid_logprobs"].nbytes
                                if iid
                                else arrays["candidate_ids"].nbytes + arrays["candidate_logprobs"].nbytes
                            ),
                        }
                    )
                for row in block:
                    row["same_tokens_as_tis"] = bool(np.array_equal(outputs[row["method"]], outputs["tis"]))
                captured.extend(block)
    finally:
        await engine.performance_stop_route.remote()
    result = {"route_comparison": comparisons, "capture_comparison": captured}
    write_record(root, "inference-comparison.json", result)
    return result


async def filter_capture_cost(engine, prefix, sampling, seed, repeats):
    methods = (("chosen_only", 0, 0), ("bounded128", 0, 128), ("full_vocabulary", -1, 0))
    records = []
    for repeat in range(-1, repeats):
        offset = (repeat + 1) % len(methods)
        for method, count, support_limit in methods[offset:] + methods[:offset]:
            started = time.perf_counter()
            arrays = await engine.performance_generate.remote(
                {
                    "prompt_token_ids": prefix,
                    "candidate_count": count,
                    "support_capture_limit": support_limit,
                    "sampling_params": {"max_tokens": 1, "seed": seed, **sampling},
                }
            )
            elapsed = time.perf_counter() - started
            record = {
                "method": method,
                "repeat": repeat,
                "seconds": elapsed,
                "response_tokens": len(arrays["token_ids"]),
                "candidate_payload_bytes": arrays["candidate_ids"].nbytes + arrays["candidate_logprobs"].nbytes,
            }
            if support_limit:
                complete = bool(arrays["support_complete"].all())
                record.update(support_complete=complete, captured_finite_tokens=int(arrays["support_size"].sum()))
                if complete:
                    mass = np.exp(arrays["candidate_logprobs"].astype(np.float64)).sum(-1)
                    np.testing.assert_allclose(mass, 1, atol=1e-5, rtol=1e-5)
            records.append(record)
    return records


async def dense_prefixes(trainer, base, root):
    sampled_observer = bool(trainer.cfg.generator.engine_init_kwargs.get("logits_processors"))
    repeats = 2 if trainer.cfg.trainer.placement.policy_num_nodes == 1 else 5
    layout = (await asyncio.gather(*trainer.policy_model.async_run_ray_method("pass_through", "qualification_layout")))[
        0
    ]
    vocabulary = layout["vocabulary"]
    lengths = base["response_mask"].sum(-1)
    selected_rows = sorted(range(len(lengths)), key=lambda row: int(lengths[row]), reverse=True)[:4]
    positions = torch.zeros((len(lengths), 3), dtype=torch.long)
    selected = torch.zeros_like(positions, dtype=torch.bool)
    requests = []
    for row in selected_rows:
        valid = base["loss_mask"][row].bool().nonzero().flatten()
        if not len(valid):
            continue
        chosen = valid[torch.linspace(0, len(valid) - 1, min(3, len(valid))).round().long()].unique()
        tokens = base["sequences"][row][base["attention_mask"][row].bool()].tolist()
        for slot, position in enumerate(chosen.tolist()):
            positions[row, slot], selected[row, slot] = position, True
            requests.append(
                {
                    "row": row,
                    "slot": slot,
                    "position": position,
                    "response_length": int(lengths[row]),
                    "prefix": tokens[: len(tokens) - int(lengths[row]) + position],
                }
            )
    batch = copy_batch(base, [key for key in base if not key.startswith("score_")])
    batch.update(
        {
            "score_topk_indices": (
                torch.arange(vocabulary, dtype=torch.int32)
                .expand(len(lengths), 3, -1)
                .clone()
                .masked_fill(~selected[..., None], -1)
            ),
            "score_probe_positions": positions,
        }
    )
    await choose_arm(trainer, "tis")
    trainer.policy_model.backload_to_gpu()
    outputs = await asyncio.gather(*trainer.policy_model.async_run_ray_method("mesh", "forward", batch))
    scored = concatenate_outputs_after_mesh_dispatch(trainer.policy_model.actor_infos, outputs)["score_old_logprobs"]
    await trainer.inference_engine_client.wake_up()
    engine = trainer.inference_engine_client.engines[0].inference_engine_actor
    reports = []
    for index, request in enumerate(requests):
        arrays = await engine.performance_generate.remote(
            {
                "prompt_token_ids": request["prefix"],
                "candidate_count": -1,
                "sampled8": sampled_observer,
                "sampling_params": {"max_tokens": 1, "seed": 97 + index},
            }
        )
        q = np.empty(vocabulary, dtype=np.float32)
        q[arrays["candidate_ids"][0]] = arrays["candidate_logprobs"][0]
        if not sampled_observer:
            probabilities = np.exp(q.astype(np.float64))
            probabilities /= probabilities.sum()
            draws = np.random.default_rng(97 + index).choice(vocabulary, size=8, replace=True, p=probabilities)
            arrays["iid_indices"] = draws.astype(np.int32)[None]
            arrays["iid_logprobs"] = q[draws][None]
        np.testing.assert_allclose(q[arrays["iid_indices"][0]], arrays["iid_logprobs"][0], atol=1e-5, rtol=1e-5)
        oracle = {
            "current_logprobs": scored[request["row"], request["slot"]].numpy(),
            "behavior_logprobs": q,
            "iid_indices": arrays["iid_indices"][0],
            "iid_logprobs": arrays["iid_logprobs"][0],
            "prefix_token_ids": np.asarray(request["prefix"], dtype=np.int32),
        }
        record = {key: value for key, value in request.items() if key != "prefix"}
        record["iid_capture"] = "v1_gpu_observer" if sampled_observer else "cpu_full_vocabulary_reference"
        record["prefix_length"] = len(request["prefix"])
        record["arrays"] = write_arrays(root, f"dense-prefix-{index}.npz", oracle)
        record["filters"] = []
        for sampling in ({"top_k": 32}, {"top_k": 32, "top_p": 0.9}, {"top_p": 0.9}, {"top_p": 0.99}):
            filtered = await engine.performance_generate.remote(
                {
                    "prompt_token_ids": request["prefix"],
                    "candidate_count": -1,
                    "sampling_params": {"max_tokens": 1, "seed": 97 + index, **sampling},
                }
            )
            values = np.empty(vocabulary, dtype=np.float32)
            values[filtered["candidate_ids"][0]] = filtered["candidate_logprobs"][0]
            array_ref = write_arrays(
                root, f"dense-prefix-{index}-filter-{len(record['filters'])}.npz", {"behavior_logprobs": values}
            )
            record["filters"].append(
                {
                    "sampling": sampling,
                    "support": int(np.isfinite(values).sum()),
                    "arrays": array_ref,
                    "capture_comparison": await filter_capture_cost(
                        engine, request["prefix"], sampling, 97 + index, repeats
                    ),
                }
            )
        reports.append(record)
        write_record(root, "dense-prefixes.json", reports)
    return reports


@ray.remote(num_cpus=1, max_retries=0, max_calls=1)
def performance_entrypoint(cfg):
    smoke = cfg.trainer.placement.policy_num_nodes == 1
    planned = schedule(2 if smoke else 5)
    assert cfg.trainer.max_steps == len(planned)
    before = set(placement_group_table())
    exp = BasePPOExp(cfg)
    exp.get_worker_classes = lambda: (PolicyWorker, CriticWorker, RefWorker)
    root = cfg.trainer.ckpt_path.rstrip("/") + "/performance"
    trainer, observers = None, []
    with exp.get_tracker() as tracker:
        try:
            trainer = exp._setup_trainer(tracker)
            inference_actors = [engine.inference_engine_actor for engine in trainer.inference_engine_client.engines]
            engine_runtime = ray.get([engine.performance_info.remote() for engine in inference_actors])
            assert all("39e62869693c" in engine["vllm_version"] for engine in engine_runtime), engine_runtime
            write_record(root, "engine-runtime.json", engine_runtime)
            observers = [
                DeviceMemory.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)
                ).remote()
                for node in ray.nodes()
                if node["Alive"] and node["Resources"].get("GPU", 0) > 0
            ]
            live, starts, saved, profiles = [], {}, {}, []
            original_publish, original_log = trainer._publish, trainer._log_metrics_stdout
            original_training, original_finish = trainer._run_training, trainer._finish_score_probe
            original_finalize = trainer._finalize_training

            async def publish(step):
                repeat, arm = planned[step - 1]
                await choose_arm(trainer, arm)
                if repeat == -1 and arm == "sc128":
                    await asyncio.gather(*(engine.performance_start_cpu_profile.remote() for engine in inference_actors))
                await reset_memory(trainer, observers, f"live-{step}")
                starts[step] = time.perf_counter()
                await original_publish(step)

            async def training(batch, *, step_wall=None):
                arm = planned[trainer.global_step - 1][1]
                if arm == "sc128" and "base" not in saved:
                    saved["base"] = TrainingInputBatch(dict(batch.items()), routed_expert_rows=batch.routed_expert_rows)
                    saved["base"].metadata = copy.deepcopy(batch.metadata)
                return await original_training(batch, step_wall=step_wall)

            def finish(batch):
                original_finish(batch)
                assert not any(key.startswith("score_") for key in batch)

            def log(payload, step, kind="train"):
                if kind == "train":
                    elapsed = time.perf_counter() - starts[step]
                    repeat, arm = planned[step - 1]
                    memory = {
                        "ranks": ray.get(trainer.policy_model.async_run_ray_method("pass_through", "performance_peaks")),
                        "devices": ray.get([observer.report.remote(f"live-{step}") for observer in observers]),
                    }
                    live.append(
                        {
                            "step": step,
                            "arm": arm,
                            "repeat": repeat,
                            "seconds": elapsed,
                            "metrics": dict(payload),
                            "memory": memory,
                        }
                    )
                    if repeat == -1 and arm == "sc128":
                        profiles.extend(
                            ray.get([engine.performance_stop_cpu_profile.remote() for engine in inference_actors])
                        )
                        write_record(root, "cpu-capture-profiles.json", profiles)
                    write_record(root, "live-cycles.json", live)
                    print(
                        "SC_PERF_CYCLE",
                        step,
                        arm,
                        repeat,
                        elapsed,
                        payload.get("score_centering_probe/sampled_tokens", 0),
                        flush=True,
                    )
                return original_log(payload, step, kind)

            async def finalize(*args, **kwargs):
                await original_finalize(*args, **kwargs)
                await trainer.context.close()
                write_record(root, "cpu-capture-profiles.json", profiles)
                updates = await asyncio.gather(
                    *trainer.policy_model.async_run_ray_method("pass_through", "qualification_report")
                )
                assert all(
                    len(rank) == len(planned)
                    and all(math.isfinite(row["status"]["raw_grad_norm"]) for row in rank)
                    and any(row["status"]["raw_grad_norm"] > 0 for row in rank)
                    for rank in updates
                )
                for arm in ("probe128", "probe32"):
                    assert (
                        sum(
                            row["metrics"].get("score_centering_probe/sampled_tokens", 0)
                            for row in live
                            if row["arm"] == arm and row["repeat"] >= 0
                        )
                        > 0
                    )
                write_record(root, "live-updates.json", updates)
                await matched_training(trainer, saved["base"], observers, root, smoke)
                await inference_comparison(trainer, root, smoke)
                await dense_prefixes(trainer, saved["base"], root)
                lengths = saved["base"]["response_mask"].sum(-1)
                row = int(lengths.argmax())
                source = saved["base"]["sequences"][row][saved["base"]["attention_mask"][row].bool()].tolist()
                length, actions = (256, 128) if smoke else (102400, 49152)
                tokens = (source * ((length + len(source) - 1) // len(source)))[:length]
                reference = reference_forward(trainer, tokens, actions)
                trainer.policy_model.backload_to_gpu()
                for count in (128, 32):
                    reports = await asyncio.gather(
                        *trainer.policy_model.async_run_ray_method(
                            "pass_through",
                            "qualification_benchmark",
                            tokens,
                            actions,
                            reference,
                            "long" if not smoke else "smoke",
                            1,
                            2 if smoke else 5,
                            count,
                        )
                    )
                    write_record(root, f"long-k{count}.json", reports)
                write_record(
                    root,
                    "completed.json",
                    {
                        "steps": len(live),
                        "ranks": len(updates),
                        "config": public_config(OmegaConf.to_container(cfg, resolve=True)),
                        "weight_lr": 0.0,
                        "engine_runtime": engine_runtime,
                        "limits": [
                            "Serialized fixed-workload cycles; not asynchronous throughput or a learning comparison.",
                            "Learning rate zero and zero-reward group filtering disabled for fixed prompt workloads.",
                            (
                                "IID samples without the V1 observer are CPU full-vocabulary references, "
                                "not a cheap capture implementation. The observer changes the model runner."
                            ),
                            (
                                "Dense q is recaptured in one-step requests at generated prefixes; "
                                "it is not assumed equal to the original decode distribution."
                            ),
                            (
                                "The numeric route returns tokens and probabilities; "
                                "it does not replace native chat parsing or continuation."
                            ),
                            "Physical GPU memory is sampled every 50 ms; Torch peaks cover policy-process tensors.",
                        ],
                    },
                )

            trainer._publish, trainer._log_metrics_stdout = publish, log
            trainer._run_training, trainer._finish_score_probe = training, finish
            trainer._finalize_training = finalize
            asyncio.run(trainer.train())
        finally:
            for observer in observers:
                ray.get(observer.close.remote())
                ray.kill(observer)
            if trainer is not None:
                asyncio.run(trainer.shutdown())
            for group in set(placement_group_table()) - before:
                remove_placement_group(PlacementGroup(PlacementGroupID.from_hex(group)))
