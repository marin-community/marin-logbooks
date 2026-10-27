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

import jax
from fray.cluster import ResourceConfig
from levanter.cutlass_kernel_cache import cutlass_kernel_cache, install
from levanter.distributed import DistributedConfig
from levanter.grug.sharding import compact_grug_mesh
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe_hero_ep.attention_mitigation_diagnostic import _host_arrays, _native_smoke, _panel_record
from experiments.grug.moe_hero_ep.attention_mitigation_replay import REPLAY_GATE_NAMES
from experiments.grug.moe_hero_ep.model import OFFLOAD_CARRY_REMAT_MODE
from experiments.grug.moe_hero_ep.train import RAGGED_MOE_IMPLEMENTATION, _apply_hero_ep_runtime_defaults

logger = logging.getLogger(__name__)
TRACEBACK_INTERVAL = 120


def _record_phase(
    root: StoragePath, records: list[dict], phase: str, *, length: int, heads: int, started: float
) -> None:
    records.append(
        {
            "phase": phase,
            "time_utc": datetime.now(UTC).isoformat(),
            "elapsed": time.monotonic() - started,
        }
    )
    logger.info("NATIVE_ACCEPTANCE_PHASE length=%d heads=%d phase=%s", length, heads, phase)
    (root / f"smoke-{length}-{heads}-phases.json").write_text(json.dumps(records, allow_nan=False))


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
        root = StoragePath(request["output_root"])
        with jax.set_mesh(compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)):
            for length, heads in ((128, 4), (256, 8)):
                phases = []
                started = time.monotonic()

                phase = partial(_record_phase, root, phases, length=length, heads=heads, started=started)
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
            json.dumps({"source_revision": request["source_revision"], "cases": results}, allow_nan=False)
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
