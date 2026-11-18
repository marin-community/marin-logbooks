# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize retained native exports and reuse the campaign's phase accounting."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path

DIRECTORY = Path(__file__).parent
PHASE_COSTS = importlib.import_module("logbooks.57-inference-perfmax.analysis.hero-fp8-01a1ec55.cost_ledger").phase_costs
SUMMARY_KEYS = {
    "status",
    "reason",
    "error",
    "positions",
    "statistics",
    "limits",
    "groups",
    "cache_alignment",
    "retention_interval",
    "same_split",
    "different_split",
    "matched_repeat",
    "cached_tokens",
    "prefill_reuse",
    "decode_reuse",
    "replayed_decode_reuse",
    "settled_branches",
}


def compact(value):
    if not isinstance(value, dict):
        return value
    return {
        key: (
            item
            if key in {"statistics", "limits"}
            else {name: compact(group) for name, group in item.items()} if key == "groups" else compact(item)
        )
        for key, item in value.items()
        if key in SUMMARY_KEYS
    }


def summarize(name: str) -> None:
    result = json.loads((DIRECTORY / f"{name}-result.json").read_text())
    native = Path("scratch") / f"{name}-native"
    export_path = native / "aiperf_artifacts/profile_export_aiperf.json"
    export = json.loads(export_path.read_text())
    measurement = result["measurement"]
    leader = next(attempt["worker"] for attempt in result["attempts"] if attempt["worker"]["task_index"] == 0)
    numerical = next(check for check in measurement["checks"] if check["kind"] == "numerical")
    check_path = native / numerical["report"]
    check_hash = hashlib.file_digest(check_path.open("rb"), "sha256").hexdigest()
    if check_hash != numerical["report_sha256"]:
        raise ValueError("The retained numerical report differs from the native result")
    raw_check = json.loads(check_path.read_text())
    check_summary = {
        "status": raw_check["status"],
        "runtime_source": raw_check["runtime_source"],
        "policy": raw_check["policy"],
        "raw_report_sha256": check_hash,
        "checks": {key: compact(value) for key, value in raw_check["checks"].items()},
    }
    # Keep every selected statistics map intact. Recursive key filtering must
    # never discard its numeric fields.
    for key in ("batch", "decode"):
        if check_summary["checks"][key]["statistics"] != raw_check["checks"][key]["statistics"]:
            raise ValueError("A raw numerical statistics map was lost")
    for group in raw_check["checks"]["reference"]["groups"]:
        if (
            check_summary["checks"]["reference"]["groups"][group]["statistics"]
            != raw_check["checks"]["reference"]["groups"][group]["statistics"]
        ):
            raise ValueError("A raw reference statistics map was lost")
    (DIRECTORY / f"{name}-check-summary.json").write_text(json.dumps(check_summary, indent=2) + "\n")
    attempts = []
    for attempt in result["attempts"]:
        timeline = leader if attempt["attempt_uid"] in leader["gang_attempts"].values() else None
        phases = PHASE_COSTS(attempt, export, timeline)
        if abs(sum(phases.values()) - attempt["gpu_seconds"]) > 1e-5:
            raise ValueError("Phase accounting does not conserve allocated GPU time")
        attempts.append(
            {
                "task_id": attempt["task_id"],
                "attempt_uid": attempt["attempt_uid"],
                "node": attempt["node_name"],
                "allocation": attempt["allocation"],
                "gpu_seconds": attempt["gpu_seconds"],
                "phases": phases,
                "host_memory": attempt["worker"]["host_memory"],
            }
        )
    summary = {
        "trial_uri": result["trial_uri"],
        "scope": "Retained native one-hour exports; original bytes remain in the native archive",
        "validity": measurement["validity"],
        "serving": result["manifest"]["series"]["profile"],
        "native_export_sha256": hashlib.file_digest(export_path.open("rb"), "sha256").hexdigest(),
        "native_export": export,
        "processor_request_metrics": measurement["aggregate"]["request_metrics"],
        "attempts": attempts,
        "gpu_seconds": result["gpu_seconds"],
        "gpu_hours": result["gpu_seconds"] / 3600,
        "cost_complete": result["cost_complete"],
        "cost_basis": result["cost_basis"],
        "phase_note": (
            "Leader timestamps apply to followers only for the same retained native gang. "
            "Warmup and native drain/export remain separate from the accepted-request span."
        ),
        "processor_topology_note": (
            "Use the resolved serving profile and native deployment for TP/DP/EP and host count. "
            "Processor tp and is_multinode fields follow its pool-normalization conventions."
        ),
    }
    (DIRECTORY / f"{name}-native-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        name,
        {
            "gpu_hours": summary["gpu_hours"],
            "phase_gpu_seconds": {key: sum(a["phases"][key] for a in attempts) for key in attempts[0]["phases"]},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    for trial in parser.parse_args().names:
        summarize(trial)
