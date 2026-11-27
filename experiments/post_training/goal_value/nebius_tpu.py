# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the packed Nebius windows with the frozen Levanter Qwen3 encoder."""

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from transformers import AutoConfig

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
BLOCK = 8192


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("IRIS_OUTPUT_DIR", "results")))
    args = parser.parse_args()
    devices = jax.local_devices()
    if not devices or any(device.platform != "tpu" for device in devices):
        raise RuntimeError("expected local TPU devices")
    print(f"TPU devices: {len(devices)}", flush=True)
    payload = np.load(args.payload)
    ids = payload["ids"]
    indices = payload["indices"]
    id_offsets = payload["id_offsets"]
    index_offsets = payload["index_offsets"]
    weight = payload["weight"]
    bias = payload["bias"]
    groups = defaultdict(list)
    for i, length in enumerate(np.diff(id_offsets)):
        groups[int((length + BLOCK - 1) // BLOCK * BLOCK)].append(i)
    scores_per_window = np.diff(index_offsets)
    take_count = int(scores_per_window.max())

    started = time.perf_counter()
    snapshot = snapshot_download(MODEL, revision=REVISION, allow_patterns=["*.safetensors", "*.json"])
    hf_config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    config = replace(Qwen3Config.from_hf_config(hf_config), attn_backend=AttentionBackend.SPLASH)
    cpu = jax.devices("cpu")[0]
    cpu_mesh = jax.sharding.Mesh(np.asarray([cpu]), ("data",))
    with jax.set_mesh(cpu_mesh), jax.default_device(cpu), hax.axis_mapping({}):
        model = config.hf_checkpoint_converter(ref_checkpoint=snapshot).load_pretrained(
            Qwen3LMHeadModel, config=config, dtype=jnp.bfloat16
        )
    weights, skeleton = eqx.partition(model, eqx.is_array)
    del model
    mesh = jax.sharding.Mesh(np.asarray(devices), ("data",))
    batch_axis = hax.Axis("batch", len(devices))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    by_batch = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None))
    weights = jax.device_put(weights, replicated)
    jax.block_until_ready(weights)
    print(f"Loaded model in {time.perf_counter() - started:.1f}s", flush=True)

    @jax.jit
    def forward(parameters, tokens, positions):
        lm = eqx.combine(parameters, skeleton)
        pos = hax.Axis("position", tokens.shape[1])
        states = lm.activations(hax.named(tokens, (batch_axis, pos)), AttentionMask.causal())
        hidden = states.rearrange((batch_axis, pos, config.Embed)).array
        return jnp.take_along_axis(hidden, positions[:, :, None], axis=1).astype(jnp.bfloat16)

    results = np.full((len(scores_per_window), take_count), np.nan, dtype=np.float32)
    processed = 0
    padded_tokens = 0
    scoring_started = time.perf_counter()
    for width, group in sorted(groups.items()):
        print(f"Starting window width {width}, {len(group)} windows", flush=True)
        for start in range(0, len(group), len(devices)):
            windows = group[start : start + len(devices)]
            padded = np.zeros((len(devices), width), dtype=np.int32)
            positions = np.zeros((len(devices), take_count), dtype=np.int32)
            for slot, window in enumerate(windows):
                tokens = ids[id_offsets[window] : id_offsets[window + 1]]
                selected = indices[index_offsets[window] : index_offsets[window + 1]]
                padded[slot, : len(tokens)] = tokens
                positions[slot, : len(selected)] = selected
            with jax.set_mesh(mesh), hax.axis_mapping({"batch": "data"}):
                states = forward(weights, jax.device_put(padded, by_batch), jax.device_put(positions, by_batch))
            logits = np.asarray(states, dtype=np.float32) @ weight + bias
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            for slot, window in enumerate(windows):
                count = scores_per_window[window]
                results[window, :count] = probabilities[slot, :count]
            processed += len(windows)
            padded_tokens += len(devices) * width
            if start == 0 or (start // len(devices) + 1) % 20 == 0:
                print(f"Scored {processed}/{len(results)} windows", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "nebius-tpu-values.npy", results)
    telemetry = {
        "hardware": "TPU",
        "windows": processed,
        "input_tokens": len(ids),
        "padded_tokens": padded_tokens,
        "seconds": time.perf_counter() - scoring_started,
    }
    (args.output / "nebius-tpu-telemetry.json").write_text(json.dumps(telemetry) + "\n")
    print("NEBIUS_TPU_COMPLETE " + json.dumps(telemetry), flush=True)


if __name__ == "__main__":
    main()
