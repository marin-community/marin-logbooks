# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare experimental attention-backward mitigations on frozen Hero weights."""

import argparse
import gc
import json
import logging
import os
import statistics
import time
from functools import partial

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from fray.cluster import ResourceConfig
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P
from levanter.cutlass_kernel_cache import cutlass_kernel_cache, install
from levanter.data.text.datasets import LmDataConfig
from levanter.distributed import DistributedConfig
from levanter.grug.attention._fa4_cute_backend import (
    segmented_flash_attention_backward,
    segmented_flash_attention_forward,
)
from levanter.grug.attention._fa4_cute_config import sm100_flash4_cute_kernel_config
from levanter.grug.sharding import compact_grug_mesh
from levanter.schedule import BatchSchedule
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe_hero_ep.attention_mitigation_replay import (
    REPLAY_GATE_NAMES,
    REQUIRED_ZERO_GATE_NAMES,
    replay_attention_mitigations,
    smoke_local_replay,
)
from experiments.grug.moe_hero_ep.attention_mitigation_saved_output import (
    saved_output_backward,
    saved_output_forward,
    shadow_saved_output_forward,
)
from experiments.grug.moe_hero_ep.attention_softmax_capture import (
    AttentionCapture,
    block_forward,
    block_hidden_vjp,
    selected_attention_vjp,
)
from experiments.grug.moe_hero_ep.attention_softmax_diagnostic import (
    COMPUTE_POLICY,
    EVENT_KEYS,
    _embed,
    _final_hidden_gradient,
    _layer_masks,
    summarize_panels,
)
from experiments.grug.moe_hero_ep.attention_softmax_reference import attention_reference, compare_attention_gradients
from experiments.grug.moe_hero_ep.attention_softmax_replay import BATCH_AXES
from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.train import (
    _apply_hero_ep_runtime_defaults,
    build_train_dataset,
    build_train_loader,
    verify_ragged_pjrt,
)
from experiments.grug.moe_hero_ep.weights import restore_weights

logger = logging.getLogger(__name__)
MAX_RECORD_BYTES = 1_048_576


def _gather_records(record: dict) -> list[dict]:
    encoded = json.dumps(record, allow_nan=False).encode()
    if len(encoded) > MAX_RECORD_BYTES - 4:
        raise ValueError(f"Mitigation panel exceeds bounded JSON gather: {len(encoded)}")
    buffer = np.zeros(MAX_RECORD_BYTES, dtype=np.uint8)
    buffer[:4] = np.frombuffer(len(encoded).to_bytes(4, "little"), dtype=np.uint8)
    buffer[4 : 4 + len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    gathered = np.asarray(multihost_utils.process_allgather(buffer, tiled=False)).reshape(-1, MAX_RECORD_BYTES)
    if jax.process_index() != 0:
        return []
    return [json.loads(row[4 : 4 + int.from_bytes(row[:4].tobytes(), "little")].tobytes()) for row in gathered]


def _host_arrays(tree):
    return jax.tree.map(lambda value: np.asarray(value.addressable_shards[0].data), tree)


def _panel_record(replay, *, layer_index: int, global_sequence_index: int) -> dict:
    host = _host_arrays(replay)
    gates = dict(zip(REPLAY_GATE_NAMES, host.gates.reshape(-1).tolist(), strict=True))
    for name in REQUIRED_ZERO_GATE_NAMES:
        if gates[name] != 0:
            raise ValueError(
                f"Mitigation parity gate failed at layer {layer_index}: {name}={gates[name]}; all gates={gates}"
            )
    key_positions = np.arange(host.k.shape[1])[None, None, :]
    mask = (
        host.valid[..., None]
        & (key_positions >= host.lower_bounds[..., None])
        & (key_positions <= host.positions[..., None])
    )
    scale = host.q.shape[-1] ** -0.5
    reference = attention_reference(host.q, host.k, host.v, host.dout, mask, softmax_scale=scale)
    variants = {}
    for name, arrays in host.variants.items():
        record = compare_attention_gradients(
            reference,
            arrays.output,
            arrays.dq,
            arrays.dk,
            q=host.q,
            k=host.k,
            v=host.v,
            dout=host.dout,
            softmax_scale=scale,
            fused_dv=arrays.dv,
        )
        record.update(
            replay_output_max_abs_error=gates["baseline_forward_max_abs_error"],
            tail_output_max_abs_error=gates["capture_tail_max_abs_error"],
            tail_output_reference_rms=0.0,
            selection_fallback_count=int(host.selection_fallback_count.sum()),
        )
        for key in EVENT_KEYS:
            for event in record[key]:
                event.update(
                    rank=jax.process_index(),
                    layer=layer_index,
                    global_sequence_index=global_sequence_index,
                    query_position=int(host.positions[event["batch"], event["query"]]),
                    query_head=event["head"],
                    boundary_kv_head=event["head"] // (host.q.shape[2] // host.k.shape[2]),
                )
        variants[name] = record
    return {
        "rank": jax.process_index(),
        "layer": layer_index,
        "global_sequence_index": global_sequence_index,
        "positions": host.positions.tolist(),
        "local_row": int(host.local_row.reshape(-1)[0]),
        "gates": gates,
        "dout_l2": float(np.linalg.norm(host.dout.astype(np.float64))),
        "variants": variants,
        "projection": {
            "row_count": int(host.row_mass.size),
            "absolute_row_mass_sum": float(np.abs(host.row_mass).sum(dtype=np.float64)),
            "maximum_absolute_row_mass": float(np.abs(host.row_mass).max()),
            "minimum_probability_mass": float(host.probability_mass.min()),
            "maximum_probability_mass": float(host.probability_mass.max()),
            "maximum_absolute_correction": float(np.abs(host.correction).max()),
            "maximum_absolute_algebraic_projected_row_mass": float(np.abs(host.post_projection_mass).max()),
        },
    }


def _capture_specs():
    arrays = P(BATCH_AXES, None, None, None)
    metadata = P(BATCH_AXES, None)
    return AttentionCapture(arrays, arrays, arrays, arrays, arrays, metadata, metadata, P(), P())


@eqx.filter_jit
def _dense_pair(capture: AttentionCapture, *, implementation: str, length: int):
    def local(local_capture):
        q, k, v, dout = (
            local_capture.q[:, :length],
            local_capture.k[:, :length],
            local_capture.v[:, :length],
            local_capture.dout[:, :length],
        )
        lower, valid = local_capture.lower_bounds[:, :length], local_capture.valid[:, :length]
        scale = q.shape[-1] ** -0.5
        config = sm100_flash4_cute_kernel_config()
        offset = jnp.zeros((1,), jnp.int32)
        if implementation == "baseline":
            output, lse = segmented_flash_attention_forward(
                q, k, v, lower, valid, softmax_scale=scale, kernel_config=config, q_offset=offset
            )
            return segmented_flash_attention_backward(
                q,
                k,
                v,
                output,
                dout,
                lse,
                lower,
                valid,
                softmax_scale=scale,
                kernel_config=config,
                q_offset=offset,
            )
        forward = shadow_saved_output_forward if implementation == "fp32_saved_output_shadow" else saved_output_forward
        _, state = forward(q, k, v, lower, valid, sm_scale=scale, kernel_config=config, q_offset=offset)
        return saved_output_backward(state, dout, sm_scale=scale, kernel_config=config)

    specs = _capture_specs()
    values = jax.tree.map(jax.sharding.reshard, capture, specs)
    return jax.shard_map(local, in_specs=(specs,), out_specs=P(BATCH_AXES), check_vma=False)(values)


def _benchmark(capture: AttentionCapture, request: dict, *, layer_index: int) -> list[dict]:
    records = []
    for length in request["benchmark_lengths"]:
        for name in ("baseline", "fp32_saved_output", "fp32_saved_output_shadow"):
            start = time.perf_counter()
            result = _dense_pair(capture, implementation=name, length=length)
            jax.block_until_ready(result)
            first_call = time.perf_counter() - start
            for _ in range(request["benchmark_warmup"]):
                result = _dense_pair(capture, implementation=name, length=length)
                jax.block_until_ready(result)
            times = []
            for _ in range(request["benchmark_repetitions"]):
                start = time.perf_counter()
                result = _dense_pair(capture, implementation=name, length=length)
                jax.block_until_ready(result)
                times.append(time.perf_counter() - start)
            del result
            local_shape = [capture.q.shape[0] // jax.process_count(), length, *capture.q.shape[2:]]
            record = {
                "kernel": "Hero full-head dense-cotangent attention forward+backward",
                "implementation": name,
                "shape": {"local_q": local_shape, "kv_heads": capture.k.shape[2]},
                "dtype": "bfloat16",
                "backend": (
                    "native SM100 CuTe + XLA saved-output correction" if name != "baseline" else "native SM100 CuTe"
                ),
                "device_type": jax.local_devices()[0].device_kind,
                "device_count": jax.device_count(),
                "block_sizes": {"forward": [128, 128], "backward": [128, 128]},
                "compile_time": first_call,
                "compile_time_definition": "first invocation including compilation and first execution",
                "steady_state_time": statistics.median(times),
                "samples": times,
                "error": None,
                "git_sha": request["source_revision"],
                "xla_flags": os.environ.get("XLA_FLAGS", ""),
                "backend_env": {key: os.environ.get(key) for key in ("JAX_ENABLE_X64", "JAX_PLATFORMS")},
                "rank": jax.process_index(),
                "layer": layer_index,
                "projection_dense_supported": False,
                "saved_output_extra_bytes_per_rank": (
                    int(np.prod(local_shape) * (4 if name.endswith("shadow") else 2)) if name != "baseline" else 0
                ),
            }
            records.append(record)
    return records


def _checkpoint_comparison(model, batch, request: dict, checkpoint: dict) -> dict:
    layer_masks = _layer_masks(model, batch)
    hidden = _embed(model, batch.tokens)
    inputs = []
    for index, (mask, is_global) in enumerate(layer_masks):
        saved = jax.device_put(hidden, hidden.sharding.with_memory_kind("pinned_host"))
        jax.block_until_ready(saved)
        inputs.append(saved)
        next_hidden = block_forward(model.stacked_blocks.get_layer(index), hidden, mask, jnp.asarray(is_global))
        jax.block_until_ready(next_hidden)
        hidden.delete()
        hidden = next_hidden
    loss, dy = _final_hidden_gradient(model, hidden, batch, request["z_loss_weight"])
    jax.block_until_ready(dy)
    hidden.delete()
    layer_results = []
    for index in reversed(range(model.config.num_layers)):
        mask, is_global = layer_masks[index]
        saved = inputs.pop()
        x = jax.device_put(saved, saved.sharding.with_memory_kind("device"))
        jax.block_until_ready(x)
        saved.delete()
        layer = model.stacked_blocks.get_layer(index)
        if index in request["layers"]:
            capture = selected_attention_vjp(layer, x, dy, mask, jnp.asarray(is_global))
            replay = replay_attention_mitigations(capture, layer_index=index, query_count=request["query_count"])
            jax.block_until_ready(replay)
            start = capture.q.addressable_shards[0].index[0].indices(capture.q.shape[0])[0]
            local_row = int(_host_arrays(replay.local_row).reshape(-1)[0])
            record = _panel_record(replay, layer_index=index, global_sequence_index=start + local_row)
            del replay
            record["benchmarks"] = (
                _benchmark(capture, request, layer_index=index)
                if index in request["benchmark_layers"] and checkpoint["step"] == request["benchmark_checkpoint"]
                else []
            )
            del capture
            panels = _gather_records(record)
            if jax.process_index() == 0:
                summary = {
                    "layer": index,
                    "is_global": is_global,
                    "panel_count": len(panels),
                    "variants": {
                        name: summarize_panels([panel["variants"][name] for panel in panels])
                        for name in panels[0]["variants"]
                    },
                    "gates_max": {name: max(panel["gates"][name] for panel in panels) for name in REPLAY_GATE_NAMES},
                    "benchmarks": [row for panel in panels for row in panel["benchmarks"]],
                }
                layer_results.append(summary)
                root = StoragePath(request["output_root"]) / f"step-{checkpoint['step']}"
                (root / f"layer-{index}-panels.json").write_text(json.dumps(panels, allow_nan=False))
                (root / f"layer-{index}-summary.json").write_text(json.dumps(summary, allow_nan=False))
                logger.info("MITIGATION_LAYER_COMPLETE step=%d layer=%d", checkpoint["step"], index)
            gc.collect()
        dx = block_hidden_vjp(layer, x, dy, mask, jnp.asarray(is_global))
        jax.block_until_ready(dx)
        x.delete()
        dy.delete()
        dy = dx
    dy.delete()
    return {
        "checkpoint": checkpoint,
        "regularized_loss": float(loss),
        "layers": sorted(layer_results, key=lambda r: r["layer"]),
    }


@partial(jax.jit, static_argnames=("length", "heads"))
def _native_smoke(*, length: int, heads: int):
    return jax.shard_map(
        lambda: smoke_local_replay(sequence_length=length, query_heads=heads), out_specs=P(), check_vma=False
    )()


def run_comparison(request: dict) -> None:
    model_config = draccus.decode(GrugModelConfig, request["model"])
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=True,
        moe_implementation=model_config.moe_implementation,
        remat_mode=model_config.remat_mode,
        processes_per_task=4,
    )
    DistributedConfig().initialize()
    configure_logging(logging.INFO if jax.process_index() == 0 else logging.WARNING)
    verify_ragged_pjrt()
    install(cutlass_kernel_cache())
    mesh = compact_grug_mesh(expert_axis_size=64, replica_axis_size=1)
    data_config = draccus.decode(LmDataConfig, request["data"])
    data_key = jax.random.split(jax.random.PRNGKey(request["seed"]), 2)[0]
    if request["data_seed"] is not None:
        data_key = jax.random.PRNGKey(request["data_seed"])
    with jax.set_mesh(mesh):
        for length, heads in ((128, 4), (256, 8)):
            logger.info("Compiling native mitigation smoke length=%d heads=%d", length, heads)
            replay = _native_smoke(length=length, heads=heads)
            jax.block_until_ready(replay)
            if jax.process_index() == 0:
                gates = dict(zip(REPLAY_GATE_NAMES, _host_arrays(replay.gates).reshape(-1).tolist(), strict=True))
                (StoragePath(request["output_root"]) / f"smoke-{length}-{heads}-gates.json").write_text(
                    json.dumps(gates, allow_nan=False)
                )
            record = _panel_record(replay, layer_index=-1, global_sequence_index=-1)
            if jax.process_index() == 0:
                (StoragePath(request["output_root"]) / f"smoke-{length}-{heads}.json").write_text(
                    json.dumps(record, allow_nan=False)
                )
            del replay
        dataset = build_train_dataset(
            data_config,
            max_seq_len=model_config.max_seq_len,
            batch_schedule=BatchSchedule(request["production_batch_size"]),
            key=data_key,
        )
        loader = build_train_loader(dataset, batch_schedule=BatchSchedule(request["batch_size"]), mesh=mesh)
        offset = request["data_reference_step"] * request["production_batch_size"]
        batch = next(iter(loader.iter_from_step(offset // request["batch_size"])))
        results = []
        for checkpoint in request["checkpoints"]:
            model = restore_weights(checkpoint["uri"], checkpoint["metadata_digest"], model_config, mesh)
            model = COMPUTE_POLICY.cast_to_compute(model)
            jax.block_until_ready(model)
            result = _checkpoint_comparison(model, batch, request, checkpoint)
            if jax.process_index() == 0:
                results.append(result)
                (StoragePath(request["output_root"]) / f"step-{checkpoint['step']}.json").write_text(
                    json.dumps(result, allow_nan=False)
                )
            jax.tree.map(lambda leaf: leaf.delete(), jax.tree.leaves(eqx.filter(model, eqx.is_array)))
            del model
            gc.collect()
        if jax.process_index() == 0:
            (StoragePath(request["output_root"]) / "results.json").write_text(
                json.dumps({"request": request, "results": results}, allow_nan=False)
            )
            logger.info("ATTENTION_MITIGATION_COMPLETE %s/results.json", request["output_root"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="Local path or private object-storage URI to resolved JSON.")
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    request = json.loads(StoragePath(args.request).read_text())
    request["source_revision"] = args.source_revision
    configure_logging(logging.INFO)
    dispatch_grug_training_run(
        run_id=request["run_id"],
        config=request,
        local_entrypoint=run_comparison,
        resources=ResourceConfig.with_gpu("GB200", count=4, cpu=120, ram="890g", disk="1t", replicas=16),
        processes_per_task=4,
        max_retries_failure=0,
        max_task_failures=0,
    )


if __name__ == "__main__":
    main()
