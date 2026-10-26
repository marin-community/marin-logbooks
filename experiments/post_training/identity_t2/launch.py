# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the native T2 plan; submit once only with a matching combined CLEAR."""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import draccus
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint

from experiments.post_training.identity_context_distill.lr0_probe import MAX_TIMEOUT_SECONDS, submit_entrypoint
from experiments.post_training.identity_context_distill.runtime_recipe import configure_precision_runtime
from experiments.post_training.identity_context_distill.sft import BASE_MODEL
from experiments.post_training.identity_context_distill.sft_launch import (
    frozen_mixture,
    lr0_pass,
    plan,
    read_remote,
    sft_config,
    sft_runtime_receipt,
    source_object_list,
)
from experiments.post_training.identity_t2.relocate import R8_SHA, R8_URI, encoded, fresh_root, sha
from experiments.post_training.identity_t2.train import bound_worker, load_inputs, t2_config


def capacity() -> dict:
    command = [
        sys.executable,
        "-c",
        "from iris.cli.main import main; main()",
        "--cluster=marin",
        "rpc",
        "controller",
        "list-peers",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=90)
    data = json.loads(result.stdout[result.stdout.index("{") :])
    for peer in data["peers"]:
        if peer["peer_id"] != "cw-rno2a":
            continue
        for backend in peer["backends"]:
            now = time.time()
            observation = now - int(backend["availability"]["observation_epoch_ms"]) / 1000
            contact = now - int(peer["last_contact_ms"]) / 1000
            nodes = [
                node["name"]
                for node in backend.get("detail", {}).get("kubernetes", {}).get("nodes", [])
                if node.get("gpu_count") == 8
                and "H100" in node.get("gpu_model", "")
                and node.get("ready")
                and node.get("schedulable")
                and node.get("running_pods") == 0
            ]
            if peer["reachable"] and 0 <= observation < 120 and 0 <= contact < 120 and len(nodes) >= 4:
                return {
                    "observed_unix": now,
                    "observation_age_seconds": observation,
                    "contact_age_seconds": contact,
                    "named_idle_whole_H100x8": sorted(nodes),
                }
    raise RuntimeError("No fresh four named idle whole H100x8 hosts; no submit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "job-name",
        "run-root",
        "freeze-root",
        "freeze-sha256",
        "runtime-receipt-sha256",
        "lr0-run-root",
        "source-object-list",
    ):
        parser.add_argument("--" + name, required=True)
    for name in ("runtime-receipt", "plan-output", "orders"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    parser.add_argument("--expect-commit")
    parser.add_argument("--combined-clear", type=Path)
    parser.add_argument("--launch-receipt", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise ValueError("Explicit timeout must be positive and no greater than the reviewed native bound")
    run_root = args.run_root.rstrip("/")
    if not run_root.startswith("s3://marin-rno2a/"):
        raise ValueError("T2 training must write in RNO")
    receipt_bytes = args.runtime_receipt.read_bytes()
    receipt = sft_runtime_receipt(receipt_bytes, args.runtime_receipt_sha256)
    frozen = load_inputs(args.freeze_root, args.freeze_sha256)
    config_bytes = read_remote(BASE_MODEL + "config.json")
    generation_bytes = read_remote(BASE_MODEL + "generation_config.json")
    data_bytes = encoded(frozen.data_config)
    config = t2_config(run_root, args.job_name, config_bytes, generation_bytes, data_bytes, receipt)
    old_bytes, _ = frozen_mixture()
    baseline = sft_config(run_root, args.job_name, config_bytes, generation_bytes, old_bytes, receipt)
    if dataclasses.replace(config, data=baseline.data) != baseline:
        raise ValueError("T2 differs from the CD16 recipe outside the approved data replacement")
    reference = receipt["routing_reference"]
    configured, observed = configure_precision_runtime(
        config,
        config_bytes=config_bytes,
        generation_bytes=generation_bytes,
        rounding_bytes=read_remote(reference["path"]),
        rounding_path=reference["path"],
        rounding_sha256=reference["sha256"],
        xla_precision=receipt["xla_precision"],
    )
    if observed != receipt or configured.data != config.data:
        raise ValueError("T2 did not reproduce the exact CD16 precision receipt")
    lr0 = lr0_pass(args.lr0_run_root, receipt)
    objects = source_object_list(args.source_object_list)
    orders = args.orders.read_bytes()
    job_plan = {
        **plan(
            args.job_name,
            run_root,
            args.timeout_seconds,
            "interactive",
            configured,
            args.runtime_receipt_sha256,
            receipt["xla_precision"],
        ),
        "orders_sha256": sha(orders),
        "baseline_tag": "snowball-identity-cd16-2026.10.26",
        "baseline_commit": "cc127d5420215679bfa45b7d0cbbfa201f1f025d",
        "all_non_data_config_equal_CD16": True,
        "actual_runtime_receipt_exact_CD16": True,
        "data_seed": configured.data_seed,
        "fresh_optimizer_updates_and_data_offset": True,
        "source_model": BASE_MODEL,
        "freeze_root": args.freeze_root,
        "freeze_sha256": args.freeze_sha256,
        "regional_files": frozen.freeze["files"],
        "first128": frozen.freeze["first128"],
        "native_checkpoint": f"{configured.trainer.checkpointer.expanded_path(args.job_name)}/step-15",
        "HF_export": f"{configured.hf_save_path}/step-15",
        "HF_label_15_means_completed_updates": 16,
        "optimizer_config": draccus.encode(configured.optimizer),
        "startup": {
            "full_regional_file_hashes_before_model_load": True,
            "source_bind_and_502_initialized_FP32_masters": True,
            "router_dot_FP32_witness": True,
            "actual_trainer_iterator_prebuffer_batches": 16,
            "actual_arrays_checked_before_each_train_step": True,
            "actual_token_mask_and_query_key_segment_fingerprints": True,
            "nonzero_or_missing_dispatch_metrics_fail_closed": True,
        },
        "export_audit": (
            "unchanged native checkpoint-before-HF, BF16+26F32, in-memory-master save audit; "
            "then regional readback before eval"
        ),
        "lr0_pass": lr0,
        "source_object_list": objects,
        "downstream": {
            "all_priority": "interactive",
            "executor": "margay",
            "after_export_audit": ["gate", "screen", "paired_math", "BFCL"],
            "panel60_executor": "vizcacha",
            "no_OPD": True,
        },
        "budget": {"soft_report_only": True, "OPD_plus_T2_flag_H100h": 300, "closed_before_train_H100h": 138.2672066667},
    }
    raw_plan = encoded(job_plan)
    if args.plan_output.exists():
        if args.plan_output.read_bytes() != raw_plan:
            raise ValueError("Existing plan differs; do not silently replace a reviewed plan")
    else:
        args.plan_output.write_bytes(raw_plan)
    print(
        json.dumps(
            {
                "plan_sha256": sha(raw_plan),
                "plan_output": str(args.plan_output),
                "clean": job_plan["clean"],
                "submit": args.run,
            }
        ),
        flush=True,
    )
    if not args.run:
        return
    if not (args.combined_clear and args.launch_receipt and args.expect_commit):
        raise ValueError("Submit needs combined CLEAR, exclusive launch receipt and reviewed commit")
    if not job_plan["clean"] or job_plan["commit"] != args.expect_commit:
        raise ValueError("Submit requires the exact clean reviewed worker pin")
    clear = json.loads(args.combined_clear.read_bytes())
    assert clear["verdict"] == "CLEAR" and clear["reviewer"] == "marten"
    assert clear["pin"] == args.expect_commit and clear["plan_sha256"] == sha(raw_plan)
    assert clear["freeze_sha256"] == args.freeze_sha256 and clear["orders_sha256"] == sha(orders)
    assert sha(Path(clear["review_path"]).read_bytes()) == clear["review_sha256"]
    assert not args.launch_receipt.exists(), "Prior invocation exists; inspect it, never duplicate"
    guard = fresh_root(run_root)
    assert sha(read_remote(R8_URI)) == R8_SHA
    slots = capacity()
    with connect_controller(cluster_name="marin") as connection:
        with IrisClient.remote(connection.url, workspace=Path.cwd(), credentials=connection.credentials) as client:
            assert not client.list_jobs(prefix="/karan/" + args.job_name), "Owned namespace already exists"
    assert args.orders.read_bytes() == orders, "Orders changed; re-read before submit"
    assert time.time() - slots["observed_unix"] < 120
    opening = {
        **job_plan,
        "state": "pre-submit",
        "submit_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "combined_clear_sha256": sha(args.combined_clear.read_bytes()),
        "capacity": slots,
        "freshness": guard,
        "final_H100h": None,
    }
    with args.launch_receipt.open("xb") as stream:
        stream.write(encoded(opening))
    entrypoint = Entrypoint.from_callable(
        bound_worker,
        run_root,
        args.job_name,
        receipt_bytes,
        args.runtime_receipt_sha256,
        args.freeze_root,
        args.freeze_sha256,
        lr0["source_receipt"],
        objects,
    )
    job_id = submit_entrypoint(args.job_name, entrypoint, args.timeout_seconds, "interactive", receipt["xla_precision"])
    args.launch_receipt.write_bytes(encoded({**opening, "state": "submitted", "job": job_id}))
    print(json.dumps({"job": job_id, "state": "submitted", "automatic_retry": False}), flush=True)


if __name__ == "__main__":
    main()
