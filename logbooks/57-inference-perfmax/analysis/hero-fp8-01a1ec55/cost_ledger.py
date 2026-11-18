# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sum retained campaign receipts, including rejected and failed allocations."""

import json
from datetime import UTC, datetime
from pathlib import Path

DIRECTORY = Path(__file__).parent
INPUTS = (
    ("Triton FP8 screen", "diagnostic-a1-result.json", "triton-a1-native"),
    ("BF16 C64 failed setup", "bf16-c64-a1-failed-result.json", None),
    ("Marlin-experts FP8 screen", "marlin-experts-a1-result.json", "marlin-experts-a1-native"),
    ("Marlin-all numerical screen", "marlin-all-a1-execution.json", None),
    ("Overlay collection failure", "selective-component-a1-failed-execution.json", None),
    ("Overlay import-binding failure", "selective-component-a2-failed-execution.json", None),
    ("Overlay input-hash failure", "selective-component-a3-failed-execution.json", None),
    ("CPU native wheel build", "selective-build-a1-execution.json", None),
    ("CPU canonical native wheel build", "marlin-canonical-build-a1-execution.json", None),
    ("CPU padding native wheel build", "padding-router-build-a1-execution.json", None),
    ("CPU hybrid native wheel build", "hybrid-build-a1-execution.json", None),
    ("Installed canonical wrapper and graph controls", "marlin-canonical-component-a1-execution.json", None),
    ("Canonical full-model cache-first rejection", "marlin-canonical-model-a1-execution.json", None),
    ("Zero-weight padded-route component", "marlin-padding-a1-execution.json", None),
    ("Fixed-padding actual-wheel component", "padding-fixed-a1-execution.json", None),
    ("Padding full-model reference rejection", "padding-model-a1-execution.json", None),
    ("Mixed Triton component configuration failure", "triton-mixed-a1-execution.json", None),
    ("Mixed Triton component context setup failure", "triton-mixed-a2-execution.json", None),
    ("Native mixed Triton loader/output/graph component", "triton-mixed-a3-execution.json", None),
    ("Hybrid ignored-layer test lifecycle failure", "hybrid-wheel-a1-execution.json", None),
    ("Hybrid20-pass tests and manual reference-context failure", "hybrid-wheel-a2-execution.json", None),
    ("24 of48 Triton FP8 model reference rejection", "triton-mixed-model-a1-execution.json", None),
    ("48 of48 selective Triton model numerical rejection", "triton-selective-model-a1-execution.json", None),
    ("Actual-wheel focused tests", "selective-wheel-tests-a1-execution.json", None),
    ("Marlin history component import failure", "marlin-history-a1-execution.json", None),
    ("Marlin order control partial writer failure", "marlin-order-a2-execution.json", None),
    ("Marlin order-only control", "marlin-order-a3-execution.json", None),
    ("Selective screen withdrawn before scheduling", "selective-fp8-marlin-a1-queue-cost.json", None),
    ("Selective full-model screen", "selective-fp8-marlin-a2-execution.json", None),
    ("BF16 C64", "bf16-c64-a2-result.json", "bf16-c64-a2-native"),
    ("BF16 C32", "bf16-c32-a1-result.json", "bf16-c32-a1-native"),
    ("BF16 C128", "bf16-c128-a1-result.json", "bf16-c128-a1-native"),
    ("Selective FP8 C128", "selective-fp8-c128-a1-result.json", "selective-fp8-c128-a1-native"),
    ("Selective FP8 C64", "selective-fp8-c64-a1-result.json", "selective-fp8-c64-a1-native"),
)


def phase_costs(attempt: dict, export: dict | None, timeline: dict | None = None) -> dict:
    allocation = attempt.get("allocation", {})
    worker = timeline or attempt.get("worker", {})
    start, end = allocation.get("scheduled_unix"), allocation.get("end_unix")
    if not export or not worker.get("ready_unix") or start is None or end is None:
        return {"unsplit_allocation_gpu_seconds": attempt.get("gpu_seconds")}
    profiling = [
        datetime.fromisoformat(export[key]).replace(tzinfo=UTC).timestamp() for key in ("start_time", "end_time")
    ]
    boundaries = [
        start,
        worker["ready_unix"],
        worker["checks_completed_unix"],
        *profiling,
        worker.get("measured_unix", end),
        end,
    ]
    names = (
        "setup",
        "checks",
        "warmup_and_harness_start",
        "profiling_and_native_grace",
        "post_request_harness_drain_and_export",
        "shutdown_and_upload",
    )
    rate = attempt["gpu_seconds"] / (end - start)
    return {
        name + "_gpu_seconds": max(0, min(end, right) - max(start, left)) * rate
        for name, left, right in zip(names, boundaries[:-1], boundaries[1:], strict=True)
    }


def main() -> None:
    rows = []
    for name, filename, cache in INPUTS:
        path = DIRECTORY / filename
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        export_path = Path("scratch") / (cache or "unused") / "aiperf_artifacts/profile_export_aiperf.json"
        export = json.loads(export_path.read_text()) if export_path.exists() else None
        attempts = record["attempts"]
        timelines = {
            uid: attempt["worker"]
            for attempt in attempts
            if attempt.get("worker", {}).get("status") == "succeeded" and attempt["worker"].get("ready_unix")
            for uid in attempt["worker"].get("gang_attempts", {}).values()
        }
        costs = [attempt.get("gpu_seconds") for attempt in attempts]
        rows.append(
            {
                "name": name,
                "receipt": filename,
                "state": record.get("job_state", record.get("state")),
                "gpu_seconds": sum(cost or 0 for cost in costs),
                "cost_complete": all(cost is not None for cost in costs),
                "cost_basis": record.get("cost_basis", "Native allocation receipts; queueing excluded"),
                "attempts": [
                    {
                        "uid": attempt.get("attempt_uid", attempt.get("uid")),
                        "gpu_seconds": attempt.get("gpu_seconds"),
                        "allocation": attempt.get("allocation"),
                        "phases": phase_costs(
                            attempt,
                            export if attempt.get("worker", {}).get("status") == "succeeded" else None,
                            timelines.get(attempt.get("attempt_uid")),
                        ),
                    }
                    for attempt in attempts
                ],
            }
        )
    total = sum(row["gpu_seconds"] for row in rows)
    output = {
        "scope": "Finished owned allocations only; pending/live work is excluded until its terminal receipt is retained",
        "gpu_type": "GB200",
        "gpu_seconds": total,
        "gpu_hours": total / 3600,
        "cost_complete_for_listed_allocations": all(row["cost_complete"] for row in rows),
        "phase_note": (
            "Profiling includes the accepted-request span and native grace. Warmup includes harness startup. "
            "Post-request harness time includes native drain/export until process exit; "
            "those activities are not separately timed. "
            "Matching follower attempts use their leader timing, verified by native gang attempt IDs. "
            "Unsplit failures and numerical/component screens retain their full allocation cost. "
            "CPU builds each requested eight CPUs for 98, 77, 99 and 78 allocated seconds, respectively, and zero GPUs."
        ),
        "rows": rows,
    }
    (DIRECTORY / "cost-ledger.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in ("gpu_seconds", "gpu_hours", "cost_complete_for_listed_allocations")}))


if __name__ == "__main__":
    main()
