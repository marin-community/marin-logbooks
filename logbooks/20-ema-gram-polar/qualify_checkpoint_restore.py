# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restore a validated pilot for infrastructure qualification, never resume training."""

import argparse
import hashlib
import json
import os
import urllib.request
import uuid
from collections import Counter
from dataclasses import replace
from pathlib import Path

import draccus
import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
from google.cloud import storage
from levanter.checkpoint import load_checkpoint
from levanter.main.train_lm import TrainLmConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer_state import TrainerState


def resolve(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Qualification request digest differs")
    request = json.loads(raw)
    proof = request["completion"]
    config = request["config"]["train_config"]
    if not proof["scientifically_complete"] or proof["stored_native_counter"] != 7630:
        raise ValueError("Require a fully validated pilot checkpoint")
    if proof["index"] != 16 or proof["source"] != "2bca5a11292d8631a0d200c5a1f9bfcc66bb2ce8":
        raise ValueError("Unexpected qualification checkpoint lineage")
    if (config["optimizer"]["alpha"], config["trainer"]["seed"], config["trainer"]["num_train_steps"]) != (
        0.25,
        0,
        7630,
    ):
        raise ValueError("Qualification uses a seed0 pilot, not a confirmation")
    expected = config["trainer"]["checkpointer"]["base_path"].rstrip("/") + "/step-7629"
    if request["checkpoint"] != expected or proof["checkpoint"]["uri"] != expected:
        raise ValueError("Checkpoint differs from frozen config and completion")
    if not expected.startswith("gs://marin-us-central2/"):
        raise ValueError("Keep full checkpoint reads in us-central2")
    cfg = draccus.decode(TrainLmConfig, config)
    cfg = replace(
        cfg,
        trainer=replace(
            cfg.trainer,
            jax_config=config["trainer"]["jax_config"],
            load_checkpoint_path=config["trainer"]["load_checkpoint_path"],
        ),
    )
    if draccus.encode(cfg) != config:
        raise ValueError("Native configuration round trip differs")
    if request["qualification_mesh_axes"] != {"data": 1, "model": 4, "replica": 1}:
        raise ValueError("Qualification requires the explicit one-host four-device mesh")
    cfg = replace(
        cfg, trainer=replace(cfg.trainer, mesh=replace(cfg.trainer.mesh, axes=request["qualification_mesh_axes"]))
    )
    return request, cfg


def template_for(request, cfg):
    vocab = hax.Axis("vocab", request["vocab_size"])
    tx = cfg.optimizer.build(cfg.trainer.num_train_steps)
    template = eqx.filter_eval_shape(
        lambda: TrainerState.init(
            tx, cfg.model.build(vocab, key=jax.random.PRNGKey(0)), key=jax.random.PRNGKey(0), mp=cfg.trainer.mp
        )
    )
    arrays = [x for x in jax.tree.leaves(template) if isinstance(x, jax.ShapeDtypeStruct)]
    actual = Counter((tuple(x.shape), str(x.dtype)) for x in arrays)
    expected = Counter((tuple(x["shape"]), x["dtype"]) for x in request["manifest"]["arrays"])
    if actual != expected:
        raise ValueError("Native full-state template shape/dtype inventory differs from checkpoint manifest")
    return template


def restore(request, cfg):
    zone_request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/zone", headers={"Metadata-Flavor": "Google"}
    )
    with urllib.request.urlopen(zone_request, timeout=10) as response:
        zone = response.read().decode().rsplit("/", 1)[-1]
    if not zone.startswith("us-central2-"):
        raise ValueError("Refuse full checkpoint reads outside its region")
    for key, value in cfg.trainer.jax_config.items():
        jax.config.update(key, value)
    if jax.process_count() != 1 or len(jax.devices()) != 4 or any(d.platform != "tpu" for d in jax.devices()):
        raise ValueError("Expected one host and four TPU devices")
    checkpoint = request["checkpoint"]
    bucket, prefix = checkpoint[5:].split("/", 1)
    client = storage.Client()
    for filename, expected in request["completion"]["checkpoint"]["files"].items():
        blob = client.bucket(bucket).blob(prefix + "/" + filename)
        raw = blob.download_as_bytes(if_generation_match=int(expected["generation"]))
        if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError("Checkpoint metadata differs from validated evidence")
    with cfg.trainer.use_device_mesh():
        template = template_for(request, cfg)
        state = load_checkpoint(
            template, checkpoint, axis_mapping=cfg.trainer.parameter_axis_mapping, allow_partial=False
        )
        jax.block_until_ready(state)
        if int(state.step) != 7630:
            raise ValueError("Restored counter differs from validated stored counter")
        arrays = [x for x in jax.tree.leaves(state) if isinstance(x, jax.Array)]
        if len(arrays) != request["completion"]["checkpoint"]["arrays"]:
            raise ValueError("Restored array count differs from validated manifest")
        for array in arrays:
            if not bool(jnp.all(jnp.isfinite(array))):
                raise ValueError("Nonfinite restored array")
        return dict(
            restored_counter=int(state.step),
            arrays=len(arrays),
            all_arrays_finite=True,
            training_key_shape=list(state.training_key.shape),
            training_key_dtype=str(state.training_key.dtype),
            model_and_optimizer_restored=True,
            allow_partial=False,
            zone=zone,
            devices=[str(d) for d in jax.devices()],
            scope=(
                "Full native TrainerState restore qualification only. No optimizer update, data-position replay, "
                "training resume or scientific reference probe."
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--mode", choices=("validate", "logging-check", "template-check", "restore"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request, cfg = resolve(args.request, args.request_sha256)
    result = dict(
        mode=args.mode,
        request_sha256=args.request_sha256,
        checkpoint=request["checkpoint"],
        source_commit=os.environ.get("GIT_COMMIT"),
        task=os.environ.get("IRIS_TASK_ID"),
        attempt_uid=os.environ.get("IRIS_ATTEMPT_UID"),
        entrypoint_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    if args.mode != "validate":
        identity = "ema-gram-restore-qualification-" + uuid.uuid4().hex[:12]
        tracker = WandbConfig(
            entity="marin-community",
            project="marin",
            id=identity,
            name=identity,
            group="ema-gram-20-restore-qualification",
            mode="online",
            resume="never",
            background=False,
            save_code=False,
        ).init(None)
        print("W&B: https://wandb.ai/marin-community/marin/runs/" + identity, flush=True)
        try:
            tracker.log_hyperparameters(dict(**result, request=request))
            if args.mode == "template-check":
                template_for(request, cfg)
                result["full_state_template_matches_manifest"] = True
            if args.mode == "restore":
                result.update(restore(request, cfg))
            tracker.log({"qualification/success": 1}, step=0)
            result.update(identity=identity, success=True)
        finally:
            tracker.finish()
    else:
        result["configuration_validated_without_kernels"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
