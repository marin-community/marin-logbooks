# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only checkpoint diagnostics for output-softmax collapse on one Hero rack."""

import argparse
import gc
import json
import logging
from pathlib import Path

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from fray.cluster import ResourceConfig
from jax.sharding import PartitionSpec as P
from levanter.cutlass_kernel_cache import cutlass_kernel_cache, install
from levanter.data.text.datasets import LmDataConfig
from levanter.distributed import DistributedConfig
from levanter.grug.sharding import compact_grug_mesh
from levanter.kernels.pallas.fused_cross_entropy_loss.xla import linear_softmax_cross_entropy_loss_xla
from levanter.schedule import BatchSchedule
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe_hero_ep.model import _CE_BLOCK_SIZES, GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.softmax_collapse_metrics import MARGIN_HISTOGRAM_EDGES, softmax_collapse_metrics
from experiments.grug.moe_hero_ep.train import (
    _apply_hero_ep_runtime_defaults,
    build_train_dataset,
    build_train_loader,
    verify_ragged_pjrt,
)
from experiments.grug.moe_hero_ep.weights import restore_weights

logger = logging.getLogger(__name__)
BATCH_AXES = ("replica_dcn", "data", "expert")
COMPUTE_POLICY = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
METRIC_TOKEN_BLOCK = 128


def _project_metrics(hidden, head, targets, weights, *, stride: int):
    """Run production CE on every token and dense diagnostics on stratified positions."""
    batch, length, width = hidden.shape
    flat = hidden.reshape(-1, width)
    labels = targets.reshape(-1)
    production_ce, production_lse = linear_softmax_cross_entropy_loss_xla(
        flat, labels, head, block_sizes=_CE_BLOCK_SIZES, dtype=jnp.float32, fast_backward=True
    )
    # Each stratum contributes a position, avoiding systematic alignment with token patterns.
    positions = jnp.arange(length // stride) * stride + jax.random.randint(
        jax.random.PRNGKey(1729), (length // stride,), 0, stride
    )
    selected = hidden[:, positions].reshape(-1, width)
    selected_targets = targets[:, positions].reshape(-1)
    selected_weights = weights[:, positions].reshape(-1)
    selected_ce = production_ce.reshape(batch, length)[:, positions].reshape(-1)
    selected_lse = production_lse.reshape(batch, length)[:, positions].reshape(-1)
    if selected.shape[0] % METRIC_TOKEN_BLOCK:
        raise ValueError("Sampled tokens per GPU must divide the metric token block")

    def block_metrics(index):
        start = index * METRIC_TOKEN_BLOCK
        states = jax.lax.dynamic_slice_in_dim(selected, start, METRIC_TOKEN_BLOCK)
        target = jax.lax.dynamic_slice_in_dim(selected_targets, start, METRIC_TOKEN_BLOCK)
        weight = jax.lax.dynamic_slice_in_dim(selected_weights, start, METRIC_TOKEN_BLOCK)
        ce = jax.lax.dynamic_slice_in_dim(selected_ce, start, METRIC_TOKEN_BLOCK)
        lse = jax.lax.dynamic_slice_in_dim(selected_lse, start, METRIC_TOKEN_BLOCK)
        logits = jnp.dot(states, head, preferred_element_type=jnp.float32)
        metrics = softmax_collapse_metrics(logits, target, weight)
        valid = (weight > 0) & jnp.isfinite(weight) & jnp.all(jnp.isfinite(logits), axis=-1)
        is_target = jnp.arange(head.shape[-1])[None, :] == target[:, None]
        tail = jnp.sum(jnp.exp(jnp.where(is_target, -jnp.inf, logits - lse[:, None])), axis=-1)
        # The streaming backward computes exp(target_logit - lse) - 1; -ce is that difference.
        target_gradient = jnp.exp(-ce) - 1
        collapsed = (ce == 0) & (target_gradient == 0) & (tail > 0)
        metrics["sampled_production_zero_ce_with_positive_tail_count"] = jnp.sum(valid & collapsed)
        metrics["sampled_production_gradient_imbalance_sum"] = jnp.sum(jnp.where(valid, target_gradient + tail, 0))
        metrics["sampled_production_ce_sum"] = jnp.sum(jnp.where(valid, ce, 0))
        return metrics

    initial = block_metrics(0)

    def add_block(index, totals):
        return jax.tree.map(jnp.add, totals, block_metrics(index))

    metrics = jax.lax.fori_loop(1, selected.shape[0] // METRIC_TOKEN_BLOCK, add_block, initial)
    valid = (weights.reshape(-1) > 0) & jnp.isfinite(production_ce) & jnp.isfinite(production_lse)
    metrics["production_valid_count"] = jnp.sum(valid)
    metrics["production_zero_ce_count"] = jnp.sum(valid & (production_ce == 0))
    metrics["production_negative_ce_count"] = jnp.sum(valid & (production_ce < 0))
    metrics["production_nonfinite_count"] = jnp.sum((weights.reshape(-1) > 0) & ~jnp.isfinite(production_ce))
    metrics["production_ce_sum"] = jnp.sum(jnp.where(valid, production_ce, 0))
    return jax.tree.map(lambda value: jax.lax.psum(value, BATCH_AXES), metrics)


@eqx.filter_jit
def checkpoint_metrics(model: Transformer, batch, *, stride: int):
    hidden, _ = model(batch.tokens, mask=batch.attn_mask)
    targets = jnp.pad(batch.tokens[:, 1:], ((0, 0), (0, 1))).astype(jnp.int32)
    head = jax.sharding.reshard(model.output_proj, P(None, None))
    hidden = jax.sharding.reshard(hidden, P(BATCH_AXES, None, None))
    targets = jax.sharding.reshard(targets, P(BATCH_AXES, None))
    weights = jax.sharding.reshard(batch.loss_weight, P(BATCH_AXES, None))
    return jax.shard_map(
        lambda h, w, t, m: _project_metrics(h, w, t, m, stride=stride),
        in_specs=(P(BATCH_AXES, None, None), P(None, None), P(BATCH_AXES, None), P(BATCH_AXES, None)),
        out_specs=P(),
        check_vma=False,
    )(hidden, head, targets, weights)


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
        dataset = build_train_dataset(
            data_config,
            max_seq_len=model_config.max_seq_len,
            batch_schedule=BatchSchedule(request["production_batch_size"]),
            key=data_key,
        )
        loader = build_train_loader(dataset, batch_schedule=BatchSchedule(request["batch_size"]), mesh=mesh)
        offset = request["data_reference_step"] * request["production_batch_size"]
        if offset % request["batch_size"]:
            raise ValueError("Reference data offset must divide the diagnostic batch size")
        logger.info("Loading fixed training-mixture batch at sequence offset %d", offset)
        batch = next(iter(loader.iter_from_step(offset // request["batch_size"])))
        results = []
        for checkpoint in request["checkpoints"]:
            logger.info("Restoring permanent checkpoint step %d", checkpoint["step"])
            model = restore_weights(checkpoint["uri"], checkpoint["metadata_digest"], model_config, mesh)
            model = COMPUTE_POLICY.cast_to_compute(model)
            jax.block_until_ready(model)
            logger.info("Scoring checkpoint step %d", checkpoint["step"])
            metrics = checkpoint_metrics(model, batch, stride=request["stride"])
            jax.block_until_ready(metrics)
            if jax.process_index() == 0:
                row = {"checkpoint": checkpoint, "metrics": jax.tree.map(lambda v: np.asarray(v).tolist(), metrics)}
                results.append(row)
                (StoragePath(request["output_root"]) / f"step-{checkpoint['step']}.json").write_text(
                    json.dumps(row, allow_nan=False)
                )
                logger.info("CHECKPOINT_RESULT %s", json.dumps(row, allow_nan=False))
            jax.tree.map(lambda leaf: leaf.delete(), jax.tree.leaves(eqx.filter(model, eqx.is_array)))
            del model
            gc.collect()
        if jax.process_index() == 0:
            result = {
                "request": request,
                "margin_histogram_edges": [str(v) for v in MARGIN_HISTOGRAM_EDGES],
                "results": results,
                "interpretation": {
                    "production": "FP32 streaming CE, production vocabulary and batch tiles, z-loss excluded",
                    "sample": (
                        "Stratified positions from one common training-mixture batch; not independent token trials"
                    ),
                    "dense": "Dense FP32 diagnostic projection may differ from production GEMM rounding",
                    "gradients": "Per-token logit CE gradients reconstructed before weighting and BF16 backward casts",
                    "routing": "One production-shaped EP64 rack, with training capacity limits and pending QB applied",
                },
            }
            (StoragePath(request["output_root"]) / "results.json").write_text(json.dumps(result, allow_nan=False))
            logger.info("DIAGNOSTIC_COMPLETE %s/results.json", request["output_root"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    configure_logging(logging.INFO)
    resources = ResourceConfig.with_gpu(
        "GB200", count=4, cpu=120, ram="890g", disk="1t", replicas=16, target_cluster="cw-us-east-08a"
    )
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
