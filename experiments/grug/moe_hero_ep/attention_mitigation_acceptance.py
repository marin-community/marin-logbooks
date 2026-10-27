# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU native acceptance for the experimental attention mitigation path."""

import argparse
import faulthandler
import importlib
import json
import logging
import time
from datetime import UTC, datetime
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from fray.cluster import ResourceConfig
from jax.sharding import PartitionSpec as P
from levanter.cutlass_kernel_cache import cutlass_kernel_cache, install
from levanter.distributed import DistributedConfig
from levanter.grug.attention._fa4_cute_backend import segmented_flash_attention_forward
from levanter.grug.attention._fa4_cute_config import sm100_flash4_cute_kernel_config
from levanter.grug.sharding import compact_grug_mesh
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe_hero_ep.attention_mitigation_diagnostic import _host_arrays, _native_smoke, _panel_record
from experiments.grug.moe_hero_ep.attention_mitigation_replay import REPLAY_GATE_NAMES
from experiments.grug.moe_hero_ep.attention_mitigation_saved_output import saved_output_forward
from experiments.grug.moe_hero_ep.model import OFFLOAD_CARRY_REMAT_MODE
from experiments.grug.moe_hero_ep.train import RAGGED_MOE_IMPLEMENTATION, _apply_hero_ep_runtime_defaults

logger = logging.getLogger(__name__)
TRACEBACK_INTERVAL = 120


class _ForwardProbeOutput(NamedTuple):
    output: jax.Array
    lse: jax.Array
    valid: jax.Array
    output_fp32: jax.Array | None


def _record_phase(
    root: StoragePath, records: list[dict], phase: str, *, case_name: str, length: int, heads: int, started: float
) -> None:
    records.append(
        {
            "phase": phase,
            "time_utc": datetime.now(UTC).isoformat(),
            "elapsed": time.monotonic() - started,
        }
    )
    logger.info("NATIVE_ACCEPTANCE_PHASE case=%s length=%d heads=%d phase=%s", case_name, length, heads, phase)
    (root / f"{case_name}-{length}-{heads}-phases.json").write_text(json.dumps(records, allow_nan=False))


@partial(jax.jit, static_argnames=("length", "heads", "mask_pattern", "implementation"))
def _saved_output_forward_probe(*, length: int, heads: int, mask_pattern: str, implementation: str):
    def local_probe():
        shape = (1, length, heads, 128)
        key_shape = (1, length, heads // 4, 128)
        positions = jnp.arange(length)
        if mask_pattern == "two_key":
            selected = jnp.asarray([31, 63, 95, 127])
            q = jnp.zeros(shape, jnp.bfloat16).at[:, selected, :, 0].set(jnp.bfloat16(10 * 128**0.5))
            k = jnp.zeros(key_shape, jnp.bfloat16).at[:, selected, :, 0].set(1)
            v = jnp.zeros(key_shape, jnp.bfloat16).at[:, selected, :, 0].set(1)
            lower = jnp.maximum(positions - 1, 0)[None]
            valid = jnp.zeros((1, length), dtype=jnp.bool_).at[:, selected].set(True)
        else:
            q = (jax.random.normal(jax.random.PRNGKey(701), shape) * 0.5).astype(jnp.bfloat16)
            k = (jax.random.normal(jax.random.PRNGKey(702), key_shape) * 0.5).astype(jnp.bfloat16)
            v = jax.random.normal(jax.random.PRNGKey(703), key_shape).astype(jnp.bfloat16)
            valid = jnp.ones((1, length), dtype=jnp.bool_)
            if mask_pattern == "packed":
                lower = jnp.maximum((positions // (length // 2)) * (length // 2), positions - 47)[None]
                valid = valid.at[:, 0].set(False)
            elif mask_pattern == "causal":
                # This forces two K tiles even after the candidate's Q tile
                # becomes 128 rows; the packed segments would each fit one tile.
                lower = jnp.zeros((1, length), dtype=jnp.int32)
            else:
                raise ValueError(f"Unknown acceptance probe mask: {mask_pattern}")
        config = sm100_flash4_cute_kernel_config()
        offset = jnp.zeros((1,), dtype=jnp.int32)
        if implementation == "baseline":
            output, lse = segmented_flash_attention_forward(
                q, k, v, lower, valid, softmax_scale=128**-0.5, kernel_config=config, q_offset=offset
            )
            return _ForwardProbeOutput(output, lse, valid, None)
        if implementation == "fp32_saved_output":
            output, state = saved_output_forward(
                q, k, v, lower, valid, sm_scale=128**-0.5, kernel_config=config, q_offset=offset
            )
            return _ForwardProbeOutput(output, state.lse, valid, state.output_fp32)
        raise ValueError(f"Unknown acceptance probe implementation: {implementation}")

    return jax.shard_map(local_probe, out_specs=P(), check_vma=False)()


def run_acceptance(request: dict) -> None:
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=True,
        moe_implementation=RAGGED_MOE_IMPLEMENTATION,
        remat_mode=OFFLOAD_CARRY_REMAT_MODE,
        processes_per_task=1,
    )
    faulthandler.dump_traceback_later(TRACEBACK_INTERVAL, repeat=True)
    try:
        DistributedConfig().initialize()
        configure_logging(logging.INFO)
        install(cutlass_kernel_cache())
        if jax.device_count() != 1:
            raise ValueError("This acceptance entry point requires exactly one GPU")
        results = []
        forward_results = []
        root = StoragePath(request["output_root"])
        with jax.set_mesh(compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)):
            for length, heads, mask_pattern in ((128, 4, "two_key"), (256, 8, "packed"), (256, 8, "causal")):
                probes = {}
                for implementation in ("baseline", "fp32_saved_output"):
                    phases = []
                    phase = partial(
                        _record_phase,
                        root,
                        phases,
                        case_name=f"forward-{mask_pattern}-{implementation}",
                        length=length,
                        heads=heads,
                        started=time.monotonic(),
                    )
                    phase("lower_begin")
                    lowered = _saved_output_forward_probe.lower(
                        length=length, heads=heads, mask_pattern=mask_pattern, implementation=implementation
                    )
                    phase("lower_complete")
                    phase("compile_begin")
                    compiled = lowered.compile()
                    phase("compile_complete")
                    phase("dispatch_begin")
                    probe = compiled()
                    phase("dispatch_complete")
                    phase("synchronize_begin")
                    jax.block_until_ready(probe)
                    phase("synchronize_complete")
                    probes[implementation] = _host_arrays(probe)
                baseline, candidate = probes["baseline"], probes["fp32_saved_output"]
                # Inspect transferred BF16 storage directly; CPU comparison
                # cannot optimize away the device's BF16 rounding boundary.
                baseline_bits = baseline.output.view(np.uint16)
                candidate_bits = candidate.output.view(np.uint16)
                baseline_bits = np.where((baseline_bits & 0x7FFF) == 0, 0, baseline_bits)
                candidate_bits = np.where((candidate_bits & 0x7FFF) == 0, 0, candidate_bits)
                gates = {
                    "public_output_bit_mismatch_count": int(np.count_nonzero(candidate_bits != baseline_bits)),
                    "fp32_saved_nonfinite_valid_count": int(
                        np.count_nonzero(candidate.valid[:, :, None, None] & ~np.isfinite(candidate.output_fp32))
                    ),
                    "fp32_lse_nonfinite_valid_count": int(
                        np.count_nonzero(candidate.valid[:, None, :] & ~np.isfinite(candidate.lse))
                    ),
                    "baseline_lse_nonfinite_valid_count": int(
                        np.count_nonzero(baseline.valid[:, None, :] & ~np.isfinite(baseline.lse))
                    ),
                }
                record = {"length": length, "query_heads": heads, "mask_pattern": mask_pattern, "gates": gates}
                (root / f"forward-{mask_pattern}-{length}-{heads}.json").write_text(json.dumps(record, allow_nan=False))
                if any(value != 0 for value in gates.values()):
                    raise ValueError(f"Saved-output forward probe failed: {record}")
                forward_results.append(record)
                logger.info(
                    "SAVED_OUTPUT_FORWARD_PROBE_COMPLETE length=%d heads=%d mask=%s", length, heads, mask_pattern
                )
            for length, heads in ((128, 4), (256, 8)):
                phases = []
                started = time.monotonic()

                phase = partial(
                    _record_phase, root, phases, case_name="smoke", length=length, heads=heads, started=started
                )
                phase("lower_begin")
                lowered = _native_smoke.lower(length=length, heads=heads)
                phase("lower_complete")
                phase("compile_begin")
                compiled = lowered.compile()
                phase("compile_complete")
                phase("dispatch_begin")
                replay = compiled()
                phase("dispatch_complete")
                phase("synchronize_begin")
                jax.block_until_ready(replay)
                phase("synchronize_complete")
                gates = dict(zip(REPLAY_GATE_NAMES, _host_arrays(replay.gates).reshape(-1).tolist(), strict=True))
                (root / f"smoke-{length}-{heads}-gates.json").write_text(json.dumps(gates, allow_nan=False))
                record = _panel_record(replay, layer_index=-1, global_sequence_index=-1)
                (root / f"smoke-{length}-{heads}.json").write_text(json.dumps(record, allow_nan=False))
                results.append({"length": length, "query_heads": heads, "gates": gates})
                logger.info("NATIVE_ACCEPTANCE_CASE_COMPLETE length=%d heads=%d", length, heads)
        (root / "results.json").write_text(
            json.dumps(
                {"source_revision": request["source_revision"], "forward_probes": forward_results, "cases": results},
                allow_nan=False,
            )
        )
        logger.info("NATIVE_MITIGATION_ACCEPTANCE_COMPLETE")
    finally:
        faulthandler.cancel_dump_traceback_later()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    # Dispatch an importable worker so its JAX objects are built on the worker,
    # rather than cloudpickling coordinator-side JIT/AOT closures from __main__.
    worker = importlib.import_module("experiments.grug.moe_hero_ep.attention_mitigation_acceptance")
    dispatch_grug_training_run(
        run_id=args.run_id,
        config={"output_root": args.output_root, "source_revision": args.source_revision},
        local_entrypoint=worker.run_acceptance,
        resources=ResourceConfig.with_gpu("GB200", count=1, cpu=24, ram="128g", disk="64g", replicas=1),
        processes_per_task=1,
        max_retries_failure=0,
        max_task_failures=0,
    )


if __name__ == "__main__":
    main()
