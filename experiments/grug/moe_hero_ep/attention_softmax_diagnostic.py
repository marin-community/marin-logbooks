# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare checkpoint-derived fused attention VJPs with a CPU FP64 reference."""

import argparse
import gc
import json
import logging
import math
from pathlib import Path

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from fray.cluster import ResourceConfig
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P
from levanter.cutlass_kernel_cache import cutlass_kernel_cache, install
from levanter.data.text.datasets import LmDataConfig
from levanter.distributed import DistributedConfig
from levanter.grug.attention import AttentionMask, fa4_cute_segment_bounds
from levanter.grug.attention._fa4_cute_backend import fa4_cute_attention_forward
from levanter.grug.attention._fa4_cute_config import sm100_flash4_cute_kernel_config
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import compact_grug_mesh
from levanter.schedule import BatchSchedule
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe_hero_ep.attention_softmax_capture import (
    block_forward,
    block_hidden_vjp,
    selected_attention_vjp,
)
from experiments.grug.moe_hero_ep.attention_softmax_reference import attention_reference, compare_attention_gradients
from experiments.grug.moe_hero_ep.attention_softmax_replay import replay_attention_panel
from experiments.grug.moe_hero_ep.model import (
    _BATCH_AXES,
    _CE_BLOCK_SIZES,
    GrugModelConfig,
    Transformer,
    _embedding_gather,
)
from experiments.grug.moe_hero_ep.train import (
    _apply_hero_ep_runtime_defaults,
    build_train_dataset,
    build_train_loader,
    verify_ragged_pjrt,
)
from experiments.grug.moe_hero_ep.weights import restore_weights

logger = logging.getLogger(__name__)
COMPUTE_POLICY = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
MAX_PANEL_JSON_BYTES = 131_072
EVENT_KEYS = ("worst_dq_events", "worst_cancellation_events", "largest_margin_events")


@eqx.filter_jit
def _synthetic_fused_arrays():
    def local_probe():
        length, heads, width = 128, 4, 128
        q = jnp.zeros((1, length, heads, width), dtype=jnp.bfloat16).at[0, -1, :, 0].set(10 * math.sqrt(width))
        k = jnp.zeros((1, length, 1, width), dtype=jnp.bfloat16).at[0, -1, 0, 0].set(1)
        v = jnp.zeros_like(k).at[0, -1, 0, 0].set(1)
        bounds = jnp.maximum(jnp.arange(length) - 1, 0)[None, :].astype(jnp.int32)
        valid = jnp.ones((1, length), dtype=jnp.bool_)
        dout = jnp.zeros_like(q).at[0, -1, :, 0].set(1)

        def fused(query, key, value):
            return fa4_cute_attention_forward(
                query,
                key,
                value,
                bounds,
                valid,
                sm_scale=1 / math.sqrt(width),
                kernel_config=sm100_flash4_cute_kernel_config(),
                q_offset=jnp.zeros((1,), dtype=jnp.int32),
            )

        output, pullback = jax.vjp(fused, q, k, v)
        dq, dk, dv = pullback(dout)
        return q[:, -1:], k, v, dout[:, -1:], output[:, -1:], dq[:, -1:], dk, dv

    return jax.shard_map(local_probe, out_specs=P(), check_vma=False)()


def _synthetic_fused_probe() -> dict:
    arrays = _synthetic_fused_arrays()
    jax.block_until_ready(arrays)
    q, k, v, dout, output, dq, dk, dv = [np.asarray(array.addressable_shards[0].data) for array in arrays]
    mask = np.zeros((1, 1, k.shape[1]), dtype=bool)
    mask[:, :, -2:] = True
    scale = 1 / math.sqrt(q.shape[-1])
    reference = attention_reference(q, k, v, dout, mask, softmax_scale=scale)
    return compare_attention_gradients(
        reference,
        output,
        dq,
        dk,
        q=q,
        k=k,
        v=v,
        dout=dout,
        softmax_scale=scale,
        fused_dv=dv,
    )


@eqx.filter_jit
def _embed(model: Transformer, tokens):
    return model.embed_gated_norm(model.embed_norm(_embedding_gather(model.token_embed, tokens)))


@eqx.filter_jit
def _final_hidden_gradient(model: Transformer, hidden, batch, z_loss_weight: float):
    targets = jnp.pad(batch.tokens[:, 1:], ((0, 0), (0, 1))).astype(jnp.int32)

    def loss(value):
        value = model.final_gated_norm(model.final_norm(value))
        return fused_linear_softmax_cross_entropy_loss(
            value,
            model.output_proj,
            targets,
            weight=batch.loss_weight,
            reduction="mean",
            logsumexp_weight=z_loss_weight,
            dtype=jnp.float32,
            implementation="xla_fast_bwd",
            block_sizes=_CE_BLOCK_SIZES,
        )

    return jax.value_and_grad(loss)(hidden)


def _layer_masks(model: Transformer, batch):
    segment_ids = batch.attn_mask.segment_ids
    if segment_ids is not None:
        q_segment_ids = jax.sharding.reshard(segment_ids[0], P(_BATCH_AXES, None))
        segment_ids = (q_segment_ids, q_segment_ids)
    mask = AttentionMask(is_causal=True, segment_ids=segment_ids)
    shape = {"batch_size": batch.tokens.shape[0], "seq_len": batch.tokens.shape[1]}
    long_bounds, valid = fa4_cute_segment_bounds(mask, **shape, sliding_window=None)
    short_bounds, _ = fa4_cute_segment_bounds(mask, **shape, sliding_window=model.config.sliding_window)
    long_bounds, short_bounds, valid = [
        jax.sharding.reshard(value, P(_BATCH_AXES, None)) for value in (long_bounds, short_bounds, valid)
    ]
    result = []
    for index in range(model.config.num_layers):
        is_global = (index + 1) % model.config.global_every == 0 or index == model.config.num_layers - 1
        result.append((mask.with_fa4_bounds(long_bounds if is_global else short_bounds, valid), is_global))
    return result


def _sum_gradient_statistics(statistics: list[dict]) -> dict:
    summed = {}
    for key in (
        "component_count",
        "actual_squared_norm",
        "reference_squared_norm",
        "error_squared_norm",
        "absolute_error_sum",
        "reference_nonzero_components",
        "zero_actual_nonzero_reference_components",
        "reference_nonzero_vectors",
        "zero_actual_nonzero_reference_vectors",
    ):
        summed[key] = sum(stat[key] for stat in statistics)
    summed["max_absolute_error"] = max(stat["max_absolute_error"] for stat in statistics)
    summed["relative_l2_error"] = (
        math.sqrt(summed["error_squared_norm"] / summed["reference_squared_norm"])
        if summed["reference_squared_norm"]
        else None
    )
    return summed


def summarize_panels(panels: list[dict]) -> dict:
    """Aggregate squared norms, counts, histograms, and bounded events over ranks."""
    result = {key: _sum_gradient_statistics([panel[key] for panel in panels]) for key in ("dq", "dk", "dv", "output")}
    rows = {}
    for key, value in panels[0]["rows"].items():
        if key == "margin_histogram_edges":
            rows[key] = value
        elif key == "margin_histogram":
            rows[key] = np.sum([panel["rows"][key] for panel in panels], axis=0).tolist()
        elif key.endswith("_max_absolute_score_gradient_error"):
            rows[key] = max(panel["rows"][key] for panel in panels)
        else:
            rows[key] = sum(panel["rows"][key] for panel in panels)
    result["rows"] = rows
    result["panel_count"] = len(panels)
    result["replay_output_max_abs_error"] = max(panel["replay_output_max_abs_error"] for panel in panels)
    result["selection_fallback_count"] = sum(panel["selection_fallback_count"] for panel in panels)
    result["tail_output_max_abs_error"] = max(panel["tail_output_max_abs_error"] for panel in panels)
    result["tail_output_reference_rms_max"] = max(panel["tail_output_reference_rms"] for panel in panels)
    result["controls"] = {}
    for name in panels[0]["controls"]:
        result["controls"][name] = {
            key: _sum_gradient_statistics([panel["controls"][name][key] for panel in panels]) for key in ("dq", "dk")
        }
        result["controls"][name]["zero_winner_nonzero_reference_count"] = sum(
            panel["controls"][name]["zero_winner_nonzero_reference_count"] for panel in panels
        )
    for key in EVENT_KEYS:
        events = [event for panel in panels for event in panel[key]]
        score = {
            "worst_dq_events": lambda event: event["dq_error_l2"],
            "worst_cancellation_events": lambda event: abs(event["reference_winner_score_gradient"]),
            "largest_margin_events": lambda event: event["margin"],
        }[key]
        result[key] = sorted(events, key=score, reverse=True)[:8]
    return result


def _gather_panel_records(record: dict) -> list[dict]:
    encoded = json.dumps(record, allow_nan=False).encode()
    if len(encoded) > MAX_PANEL_JSON_BYTES - 4:
        raise ValueError(f"Panel JSON exceeds bounded gather size: {len(encoded)}")
    buffer = np.zeros(MAX_PANEL_JSON_BYTES, dtype=np.uint8)
    buffer[:4] = np.frombuffer(len(encoded).to_bytes(4, "little"), dtype=np.uint8)
    buffer[4 : 4 + len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    gathered = np.asarray(multihost_utils.process_allgather(buffer, tiled=False)).reshape(-1, MAX_PANEL_JSON_BYTES)
    if jax.process_index() != 0:
        return []
    return [json.loads(row[4 : 4 + int.from_bytes(row[:4].tobytes(), "little")].tobytes()) for row in gathered]


def _measure_panel(capture, *, layer_index: int, query_count: int) -> dict:
    panel = replay_attention_panel(capture, layer_index=layer_index, query_count=query_count)
    jax.block_until_ready(panel)
    host = jax.tree.map(lambda value: np.asarray(value.addressable_shards[0].data), panel)
    key_positions = np.arange(host.k.shape[1])[None, None, :]
    key_mask = (
        host.valid[..., None]
        & (key_positions >= host.lower_bounds[..., None])
        & (key_positions <= host.positions[..., None])
    )
    scale = 1 / math.sqrt(host.q.shape[-1])
    reference = attention_reference(host.q, host.k, host.v, host.dout, key_mask, softmax_scale=scale)
    record = compare_attention_gradients(
        reference,
        host.output,
        host.dq,
        host.dk,
        q=host.q,
        k=host.k,
        v=host.v,
        dout=host.dout,
        softmax_scale=scale,
        fused_dv=host.dv,
    )
    sequence_start = capture.q.addressable_shards[0].index[0].indices(capture.q.shape[0])[0]
    group = int(host.kv_group.reshape(-1)[0])
    local_row = int(host.local_row.reshape(-1)[0])
    record.update(
        rank=jax.process_index(),
        layer=layer_index,
        kv_group=group,
        global_sequence_index=sequence_start + local_row,
        positions=host.positions.tolist(),
        replay_output_max_abs_error=float(host.replay_output_max_abs_error.max()),
        selection_fallback_count=int(host.selection_fallback_count.sum()),
        dout_l2=float(np.linalg.norm(host.dout.astype(np.float64))),
    )
    for key in EVENT_KEYS:
        for event in record[key]:
            event.update(
                rank=jax.process_index(),
                layer=layer_index,
                global_sequence_index=record["global_sequence_index"],
                query_position=int(host.positions[event["batch"], event["query"]]),
                query_head=group * host.q.shape[2] + event["head"],
                boundary_kv_head=group,
            )
    return record


def _checkpoint_diagnostic(model, batch, request: dict, checkpoint: dict) -> dict:
    layer_masks = _layer_masks(model, batch)
    hidden = _embed(model, batch.tokens)
    inputs = []
    for index, (mask, is_global) in enumerate(layer_masks):
        saved = jax.device_put(hidden, hidden.sharding.with_memory_kind("pinned_host"))
        jax.block_until_ready(saved)
        inputs.append(saved)
        layer = model.stacked_blocks.get_layer(index)
        next_hidden = block_forward(layer, hidden, mask, jnp.asarray(is_global))
        jax.block_until_ready(next_hidden)
        hidden.delete()
        hidden = next_hidden
        if index % 8 == 7:
            logger.info("Forward checkpoint=%d layer=%d", checkpoint["step"], index)
    loss, dy = _final_hidden_gradient(model, hidden, batch, request["z_loss_weight"])
    jax.block_until_ready(dy)
    hidden.delete()
    logger.info("Checkpoint=%d diagnostic regularized loss=%g", checkpoint["step"], float(loss))
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
            record = _measure_panel(
                capture,
                layer_index=index,
                query_count=request["query_count"],
            )
            record["tail_output_max_abs_error"] = float(capture.tail_output_max_abs_error)
            record["tail_output_reference_rms"] = float(capture.tail_output_reference_rms)
            del capture
            panels = _gather_panel_records(record)
            if jax.process_index() == 0:
                summary = summarize_panels(panels)
                summary.update(layer=index, is_global=is_global)
                layer_results.append(summary)
                root = StoragePath(request["output_root"]) / f"step-{checkpoint['step']}"
                (root / f"layer-{index}-panels.json").write_text(json.dumps(panels, allow_nan=False))
                (root / f"layer-{index}-summary.json").write_text(json.dumps(summary, allow_nan=False))
                logger.info(
                    "LAYER_RESULT checkpoint=%d layer=%d global=%s dQ_relative=%g dK_relative=%g",
                    checkpoint["step"],
                    index,
                    is_global,
                    summary["dq"]["relative_l2_error"] or 0,
                    summary["dk"]["relative_l2_error"] or 0,
                )
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


def run_diagnostic(request: dict) -> None:
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
        logger.info("Running native fused-attention synthetic control")
        synthetic = _synthetic_fused_probe()
        if jax.process_index() == 0:
            (StoragePath(request["output_root"]) / "synthetic.json").write_text(json.dumps(synthetic, allow_nan=False))
            logger.info("Synthetic native control dQ relative error=%s", synthetic["dq"]["relative_l2_error"])
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
            logger.info("Restoring attention checkpoint %d", checkpoint["step"])
            model = restore_weights(checkpoint["uri"], checkpoint["metadata_digest"], model_config, mesh)
            model = COMPUTE_POLICY.cast_to_compute(model)
            jax.block_until_ready(model)
            result = _checkpoint_diagnostic(model, batch, request, checkpoint)
            if jax.process_index() == 0:
                results.append(result)
                (StoragePath(request["output_root"]) / f"step-{checkpoint['step']}.json").write_text(
                    json.dumps(result, allow_nan=False)
                )
            jax.tree.map(lambda leaf: leaf.delete(), jax.tree.leaves(eqx.filter(model, eqx.is_array)))
            del model
            gc.collect()
        if jax.process_index() == 0:
            output = {
                "request": request,
                "synthetic": synthetic,
                "results": results,
                "interpretation": {
                    "reference": "CPU FP64 mathematical VJP on the identical BF16 inputs and cotangents",
                    "objective": "Sparse selected-query dO replay; dK sums only those cotangents and grouped heads",
                    "upstream": (
                        "Real downstream dO from frozen model layerwise reverse, with one-rack regularized mean loss"
                    ),
                    "boundary": "Expanded fused-attention KV boundary; global layers duplicate logical KV heads",
                    "controls": (
                        "Stored-output correction reconstructions are controls; fused score gradients are not exposed"
                    ),
                    "coverage": "One selected sequence and KV group per rank, four sampled positions per selected layer",
                },
            }
            (StoragePath(request["output_root"]) / "results.json").write_text(json.dumps(output, allow_nan=False))
            logger.info("ATTENTION_DIAGNOSTIC_COMPLETE %s/results.json", request["output_root"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--source-revision", required=True, help="Published archive commit shipped in this launch.")
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    request["source_revision"] = args.source_revision
    configure_logging(logging.INFO)
    resources = ResourceConfig.with_gpu("GB200", count=4, cpu=120, ram="890g", disk="1t", replicas=16)
    dispatch_grug_training_run(
        run_id=request["run_id"],
        config=request,
        local_entrypoint=run_diagnostic,
        resources=resources,
        processes_per_task=4,
        max_retries_failure=0,
        max_task_failures=0,
    )


if __name__ == "__main__":
    main()
