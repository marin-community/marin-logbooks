# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise one busy pause through the qualified sampler and rollout workers."""

import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import ray
from cloud.iris.task_runtime import prepare_policy_model, prepare_policy_tokenizer
from marinskyrl.resource_locator import RolloutModelLoading
from omegaconf import OmegaConf
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.transfer import copy
from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.rollouts.context import RolloutRequestSpec
from transformers import set_seed

from experiments.post_training.async_probe_sampler_pause import (
    CONFIG_SHA256,
    CONFIG_URI,
    DATA_SHA256,
    DATA_URI,
    OUTPUT_URI,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
)


def main():
    configure_coreweave_s3()
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / "sampler-control.json"
    record = {"config_sha256": CONFIG_SHA256, "fixed_weights": True, "training_updates": 0, "phase": "inputs"}
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    filesystem, path = filesystem_for(CONFIG_URI)
    with filesystem.open(path, "rb") as source:
        raw = source.read()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("The qualified control configuration changed")
    cfg = OmegaConf.create(json.loads(raw))
    os.environ.update({key: str(value) for key, value in cfg.extra_env.items()})
    model = prepare_policy_model(
        SimpleNamespace(
            model_source_uri=cfg.trainer.policy.model.source_uri,
            model_source_identity=cfg.trainer.policy.model.revision,
            runtime_profile="megatron",
            rollout_model_loading=RolloutModelLoading.STREAM,
        )
    )
    tokenizer = prepare_policy_tokenizer(
        SimpleNamespace(policy_tokenizer=TOKENIZER_ID, policy_tokenizer_revision=TOKENIZER_REVISION)
    )
    cfg.trainer.policy.model.path = model.local_path
    cfg.trainer.policy.model.tokenizer_path = tokenizer.local_path
    cfg.trainer.export_path = OUTPUT_URI
    data = Path("/tmp/async-probe-sampler-control/train.parquet")
    data.parent.mkdir(parents=True, exist_ok=True)
    copy(DATA_URI + "/train.parquet", str(data))
    with data.open("rb") as source:
        record["data_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    if record["data_sha256"] != DATA_SHA256:
        raise ValueError("The frozen Snowball input bytes changed")
    cfg.data.train_data = [str(data)]

    class SamplerOnly(BasePPOExp):
        def get_policy_pg(self, timeout=None):
            return None

    set_seed(1701)
    ray.init(address="local", num_cpus=128, num_gpus=8, object_store_memory=96 * 2**30, include_dashboard=False)

    async def control():
        record["phase"] = "sampler_setup"
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        experiment = SamplerOnly(cfg)
        client = experiment.create_inference_engine_client()
        runtime = await asyncio.gather(
            *(engine.inference_engine_actor.performance_info.remote() for engine in client.engines)
        )
        if not all(
            "39e62869693c" in row["vllm_version"] and row["use_v2_model_runner"] and not row["logits_processors"]
            for row in runtime
        ):
            raise ValueError("The qualified pinned vLLM V2 path changed")
        record["engine_runtime"] = runtime
        runner = experiment.get_trajectory_runner(cfg, experiment.tokenizer, client)
        await runner.startup()
        await client.pause_generation()
        await client.resume_generation(1)
        dataset = experiment.train_dataset
        indices = np.random.default_rng(1701).choice(len(dataset), size=256, replace=False).tolist()
        record["dataset_indices"] = indices
        spec = RolloutRequestSpec.from_config(cfg)
        tasks = []
        for position, index in enumerate(indices):
            prompt = dataset.collate_fn([dataset[index]])[0]
            request = spec.request(prompt, 1, probe=position == 0)
            tasks.append(asyncio.create_task(runner.run(request)))
        record["phase"] = "generation"
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        # Match the order of magnitude of the baseline's first busy sync.
        await asyncio.sleep(150)
        record["group_states_before_pause"] = dict(Counter("done" if task.done() else "pending" for task in tasks))
        for task in tasks:
            if task.done():
                task.result()
        record["phase"] = "busy_pause"
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        await client.pause_generation()
        record["busy_pause_returned"] = True
        await client.resume_generation(1)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    try:
        asyncio.run(control())
    except BaseException as error:
        record["exception"] = {"type": type(error).__qualname__, "message": str(error)}
        raise
    finally:
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        logs = Path("/tmp/ray/session_latest/logs")
        destination = output / "engine-logs"
        destination.mkdir(exist_ok=True)
        for error_log in logs.glob("worker-*.err"):
            with error_log.open(errors="replace") as source:
                prefix = source.read(2048)
            if ":actor_name:AsyncVLLMInferenceEngine" in prefix:
                for candidate in (error_log, error_log.with_suffix(".out")):
                    if candidate.is_file():
                        shutil.copy2(candidate, destination / candidate.name)
                pid = error_log.stem.rsplit("-", 1)[-1]
                for candidate in logs.glob(f"python-core-worker-*_{pid}.log"):
                    shutil.copy2(candidate, destination / candidate.name)
        shutil.copy2(logs / "raylet.out", destination / "raylet.out")
        installed = Path(importlib.util.find_spec("vllm").origin).parent
        source = output / "vllm-source"
        source.mkdir(exist_ok=True)
        for filename in ("async_llm.py", "core.py", "core_client.py", "logprobs.py", "output_processor.py"):
            shutil.copy2(installed / "v1/engine" / filename, source / filename)
        ray.shutdown()


if __name__ == "__main__":
    main()
