# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU native acceptance for the experimental attention mitigation path."""

import argparse
import json
import logging

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

logger = logging.getLogger(__name__)


def run_acceptance(request: dict) -> None:
    DistributedConfig().initialize()
    configure_logging(logging.INFO)
    install(cutlass_kernel_cache())
    if jax.device_count() != 1:
        raise ValueError("This acceptance entry point requires exactly one GPU")
    results = []
    root = StoragePath(request["output_root"])
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)):
        for length, heads in ((128, 4), (256, 8)):
            replay = _native_smoke(length=length, heads=heads)
            jax.block_until_ready(replay)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    dispatch_grug_training_run(
        run_id=args.run_id,
        config={"output_root": args.output_root, "source_revision": args.source_revision},
        local_entrypoint=run_acceptance,
        resources=ResourceConfig.with_gpu("GB200", count=1, cpu=24, ram="128g", disk="64g", replicas=1),
        processes_per_task=1,
        max_retries_failure=0,
        max_task_failures=0,
    )


if __name__ == "__main__":
    main()
