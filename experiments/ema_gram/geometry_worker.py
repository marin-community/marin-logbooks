# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restore complete saved states, then execute fresh-data geometry measurements."""

import argparse
import hashlib
import json
import os
import time
import uuid
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from google.cloud import storage
from iris.jax.init import initialize_jax
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend
from levanter.trainer_state import TrainerState
from levanter.utils.tree_utils import inference_mode
from rigging.secrets import require_secret

from experiments.ema_gram.geometry_data import checkpoint_identity, geometry_data
from experiments.ema_gram.matrix_envelope import run_matrix_envelope
from experiments.ema_gram.power_prediction import run_power_prediction
from experiments.ema_gram.probe import resolve_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--experiment", choices=("power", "envelope"), required=True)
    parser.add_argument("--stage", choices=("screen", "test", "path"), default="screen")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--path-inputs", type=Path)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--logging-check", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    spec = next(s for s in protocol["states"] if s["state"] == args.state)
    encoded = json.loads(args.configs.read_text())[spec["config_index"]]["config"]["train_config"]
    cfg = resolve_config(encoded)
    if cfg.optimizer.alpha != spec["trajectory"] or cfg.z_loss_weight != 1e-7 or cfg.trainer.train_batch_size != 8:
        raise ValueError("Configuration does not match protocol")
    identity = "ema-gram-geometry-" + uuid.uuid4().hex[:12]
    local = Path(os.environ.get("IRIS_OUTPUT_DIR", "scratch/geometry")) / identity
    local.mkdir(parents=True)
    run = wandb.init(
        entity="marin-community",
        project="marin",
        id=identity,
        name=identity,
        group="ema-gram-20-geometry-20261113",
        mode="online",
        resume="never",
        settings={"api_key": require_secret("WANDB_API_KEY")},
        config=dict(
            training=encoded,
            protocol=protocol,
            source_commit=os.environ.get("GIT_COMMIT"),
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        ),
    )
    if run.settings.mode != "online":
        raise RuntimeError("Online W&B required")
    run.log({"logging_check/success": 1}, step=0)
    print(json.dumps(dict(event="tracking_started", url=run.url)), flush=True)
    if args.logging_check:
        run.finish()
        return
    initialize_jax()
    for key, value in cfg.trainer.jax_config.items():
        jax.config.update(key, value)
    bucket_name, prefix = args.output_prefix.removeprefix("gs://").rstrip("/").split("/", 1)
    bucket = storage.Client().bucket(bucket_name)
    prefix += "/" + identity
    started, step = time.perf_counter(), 0

    def save(name, data):
        path = local / name
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        if jax.process_index() == 0:
            bucket.blob(prefix + "/" + name).upload_from_filename(str(path))

    def save_arrays(name, **arrays):
        path = local / name
        if jax.process_index() == 0:
            np.savez_compressed(path, **arrays)
            bucket.blob(prefix + "/" + name).upload_from_filename(str(path))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            record = dict(
                name=name,
                sha256=digest,
                bytes=path.stat().st_size,
                uri="gs://" + bucket_name + "/" + prefix + "/" + name,
            )
            print(json.dumps(dict(event="array_artifact", **record)), flush=True)
            return record
        return None

    def progress(data):
        nonlocal step
        step += 1
        data = dict(step=step, elapsed_seconds=time.perf_counter() - started, **data)
        with (local / "measurements.jsonl").open("a") as stream:
            stream.write(json.dumps(data, allow_nan=False) + "\n")
        run.log(data, step=step)
        print(json.dumps(data), flush=True)
        if jax.process_index() == 0:
            bucket.blob(prefix + "/measurements.jsonl").upload_from_filename(str(local / "measurements.jsonl"))

    try:
        with cfg.trainer.use_device_mesh():
            vocab = hax.partitioning.round_axis_for_partitioning(
                hax.Axis("vocab", len(cfg.data.the_tokenizer)), cfg.trainer.compute_axis_mapping
            )
            optimizer = cfg.optimizer.build(cfg.trainer.num_train_steps)
            template = eqx.filter_eval_shape(
                lambda: TrainerState.init(
                    optimizer,
                    cfg.model.build(vocab, key=jax.random.PRNGKey(0)),
                    key=jax.random.PRNGKey(0),
                    mp=cfg.trainer.mp,
                )
            )
            checkpoint_record = checkpoint_identity(spec["checkpoint"])
            if checkpoint_record["object_manifest_sha256"] != spec["checkpoint_manifest_sha256"]:
                raise ValueError("Checkpoint storage identity changed after protocol freeze")
            save("checkpoint-identity.json", checkpoint_record)
            state = load_checkpoint(
                template, spec["checkpoint"], axis_mapping=cfg.trainer.parameter_axis_mapping, allow_partial=False
            )
            if int(state.step) != spec["expected_counter"]:
                raise ValueError("Complete checkpoint restore counter mismatch")
            native_model = inference_mode(state.model, True)
            measurement_config = replace(cfg.model, attn_backend=AttentionBackend.VANILLA)
            shape = inference_mode(
                eqx.filter_eval_shape(lambda: measurement_config.build(vocab, key=jax.random.PRNGKey(0))), True
            )
            model = jax.tree.structure(shape).unflatten(jax.tree.leaves(native_model))
            if any(x.dtype != jnp.float32 for x in jax.tree.leaves(model) if eqx.is_array(x)):
                raise ValueError("Reference parameters must be FP32")
            configuration = dict(
                training=encoded,
                state=spec,
                source_commit=os.environ.get("GIT_COMMIT"),
                protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                checkpoint_restore="Full model and optimizer; fresh frozen-probe attempt, not training resume",
                host_count=jax.process_count(),
                device_count=jax.device_count(),
                wandb=run.url,
            )
            save("configuration.json", configuration)
            save("protocol.json", protocol)
            progress(dict(event="restored", counter=int(state.step)))
            panels, weights, manifest_hash = geometry_data(cfg, protocol, args.exclusions, save)
            context = dict(
                state=state,
                model=model,
                native_model=native_model,
                cfg=cfg,
                panels=panels,
                weights=weights,
                args=args,
                protocol=protocol,
                save=save,
                save_arrays=save_arrays,
                progress=progress,
                manifest_hash=manifest_hash,
            )
            if args.experiment == "power":

                result = run_power_prediction(**context)
            else:

                result = run_matrix_envelope(**context)
            save(
                "completed.json",
                dict(
                    success=True,
                    state=args.state,
                    experiment=args.experiment,
                    stage=args.stage,
                    counter=int(state.step),
                    rows=len(result["rows"]),
                    elapsed_seconds=time.perf_counter() - started,
                    output="gs://" + bucket_name + "/" + prefix,
                    wandb=run.url,
                    source_commit=os.environ.get("GIT_COMMIT"),
                    host_count=jax.process_count(),
                    device_count=jax.device_count(),
                    data_manifest_sha256=manifest_hash,
                ),
            )
    except BaseException as error:
        save("failure.json", dict(error=repr(error), wandb=run.url, elapsed_seconds=time.perf_counter() - started))
        run.finish(exit_code=1)
        raise
    else:
        run.finish()


if __name__ == "__main__":
    main()
