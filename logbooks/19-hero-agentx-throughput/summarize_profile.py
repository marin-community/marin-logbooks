# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize native Torch traces without treating overlapping GPU time as additive.

Run beside the traces in a held pod; retain raw traces separately. Kernel names
and interval unions describe the observed window. They do not measure achieved
SM occupancy or distinguish useful NCCL transfer from waiting inside a kernel.
"""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def interval_times(intervals: list[tuple[float, float, str]]) -> dict[str, float]:
    edges = []
    for start, end, category in intervals:
        edges.extend(((start, category, 1), (end, category, -1)))
    active = defaultdict(int)
    totals = defaultdict(float)
    previous = min(edge[0] for edge in edges)
    for timestamp, category, change in sorted(edges):
        communicating = active["nccl"] > 0
        computing = active["compute"] + active["memory"] > 0
        if communicating and computing:
            key = "both"
        elif communicating:
            key = "nccl_only"
        else:
            key = "compute_or_memory_only" if computing else "empty"
        totals[key] += timestamp - previous
        active[category] += change
        previous = timestamp
    result = {key + "_milliseconds": value / 1000 for key, value in totals.items()}
    result["window_milliseconds"] = (max(edge[0] for edge in edges) - min(edge[0] for edge in edges)) / 1000
    return result


def ranked(rows: dict) -> list[dict]:
    return [
        {"name": name, **values}
        for name, values in sorted(rows.items(), key=lambda item: item[1]["total_microseconds"], reverse=True)[:40]
    ]


def summarize(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        trace = json.load(stream)
    devices = defaultdict(list)
    kernels = defaultdict(lambda: {"count": 0, "total_microseconds": 0.0})
    runtime = defaultdict(lambda: {"count": 0, "total_microseconds": 0.0})
    grug_ops = defaultdict(lambda: {"count": 0, "total_microseconds": 0.0})
    annotations = defaultdict(lambda: {"count": 0, "total_microseconds": 0.0})
    steps = []
    for event in trace["traceEvents"]:
        if event.get("ph") != "X" or "dur" not in event:
            continue
        name = event["name"]
        category = event.get("cat", "")
        duration = event["dur"]
        if category in ("kernel", "gpu_memcpy", "gpu_memset"):
            if "nccl" in name.lower():
                kind = "nccl"
            else:
                kind = "memory" if category != "kernel" else "compute"
            device = str(event.get("args", {}).get("device", "unknown"))
            devices[device].append((event["ts"], event["ts"] + duration, kind))
            key = kernels[name]
            key["count"] += 1
            key["total_microseconds"] += duration
        if category in ("cuda_runtime", "cuda_driver"):
            key = runtime[name]
            key["count"] += 1
            key["total_microseconds"] += duration
        if category == "cpu_op" and (name.startswith(("vllm::", "c10d::")) or "nccl" in name.lower()):
            key = grug_ops[name]
            key["count"] += 1
            key["total_microseconds"] += duration
        if name.startswith("ProfilerStep#"):
            steps.append({"name": name, "microseconds": duration})
        elif category == "user_annotation":
            key = annotations[name]
            key["count"] += 1
            key["total_microseconds"] += duration

    return {
        "trace": path.name,
        "vllm_version": trace.get("vllm_version"),
        "device_properties": trace.get("deviceProperties"),
        "gpu_timeline": {device: interval_times(intervals) for device, intervals in devices.items()},
        "kernels_by_summed_duration": ranked(kernels),
        "cuda_runtime_by_summed_duration": ranked(runtime),
        "native_ops_by_summed_cpu_duration": ranked(grug_ops),
        "annotations_by_summed_cpu_duration": ranked(annotations),
        "profiler_steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [summarize(path) for path in args.traces]
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print("summarized", len(results), "traces", flush=True)


if __name__ == "__main__":
    main()
