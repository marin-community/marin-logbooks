# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize a sealed Nsight SQLite export using GPU interval unions.

Run after the model exits, on a host with room for the export. CUDA API and
NVTX elapsed durations can include waiting. Kernel intervals do not establish
SM occupancy, achieved bandwidth, or useful transfer time inside NCCL kernels.
"""

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from summarize_profile import interval_times, ranked

GLOBAL_THREAD_ID_BITS = 24


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def add_duration(rows: dict, name: str, duration: float) -> None:
    row = rows.setdefault(name, {"count": 0, "total_microseconds": 0.0, "max_microseconds": 0.0})
    row["count"] += 1
    row["total_microseconds"] += duration
    row["max_microseconds"] = max(row["max_microseconds"], duration)


def longest_gaps(intervals: list[tuple[float, float, str]]) -> list[dict]:
    ordered = sorted(intervals)
    origin = cursor = ordered[0][0]
    gaps = []
    for start, end, _ in ordered:
        if start > cursor:
            gaps.append(
                {"start_from_first_gpu_event_ms": (cursor - origin) / 1000, "milliseconds": (start - cursor) / 1000}
            )
        cursor = max(cursor, end)
    return sorted(gaps, key=lambda row: row["milliseconds"], reverse=True)[:20]


def api_union(intervals: list[tuple[float, float, str]]) -> dict | None:
    if not intervals:
        return None
    times = interval_times(intervals)
    return {
        "inside_traced_api_milliseconds": times.get("compute_or_memory_only_milliseconds", 0),
        "between_traced_api_milliseconds": times.get("empty_milliseconds", 0),
        "first_to_last_api_milliseconds": times["window_milliseconds"],
    }


def summarize(path: Path, range_prefix: str) -> dict:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    names = dict(connection.execute("SELECT id,value FROM StringIds"))
    processes = {row[0]: {"pid": row[1], "name": row[2]} for row in connection.execute("SELECT * FROM PROCESSES")}
    gpu_properties = {
        row[0]: {"name": row[1], "uuid": "GPU-" + row[2], "sm_count": row[3]}
        for row in connection.execute("SELECT id,name,uuid,smCount FROM TARGET_INFO_GPU")
    }
    devices = defaultdict(list)
    device_processes = defaultdict(set)
    kernels = defaultdict(dict)
    counts = defaultdict(lambda: defaultdict(int))
    for start, end, device, process, name_id in connection.execute(
        "SELECT start,end,deviceId,globalPid,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        assert end >= start
        assert process in processes
        name = names.get(name_id, str(name_id))
        kind = "nccl" if "nccl" in name.lower() else "compute"
        devices[device].append((start / 1000, end / 1000, kind))
        device_processes[device].add(process)
        counts[device]["kernel"] += 1
        counts[device][kind] += 1
        add_duration(kernels[device], name, (end - start) / 1000)
    assert devices
    memory_tables = []
    for table in ["CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"]:
        if table not in tables:
            continue
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        assert {"start", "end", "deviceId"} <= columns, (table, columns)
        memory_tables.append(table)
        for start, end, device in connection.execute(f"SELECT start,end,deviceId FROM {table}"):
            assert end >= start
            devices[device].append((start / 1000, end / 1000, "memory"))
            counts[device]["memory"] += 1
    all_processes = set().union(*device_processes.values())
    api = defaultdict(dict)
    profiler_control_api = defaultdict(dict)
    api_intervals = defaultdict(list)
    api_tables = []
    for table in ["CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"]:
        if table not in tables:
            continue
        api_tables.append(table)
        for start, end, thread, name_id in connection.execute(f"SELECT start,end,globalTid,nameId FROM {table}"):
            process = thread & ~((1 << GLOBAL_THREAD_ID_BITS) - 1)
            if process not in all_processes:
                continue
            assert process in processes
            name = names.get(name_id, str(name_id))
            if "profiler" in name.lower():
                add_duration(profiler_control_api[process], name, (end - start) / 1000)
                continue
            add_duration(api[process], name, (end - start) / 1000)
            api_intervals[process].append((start / 1000, end / 1000, "compute"))
    annotations = defaultdict(dict)
    incomplete_annotations = defaultdict(int)
    if "NVTX_EVENTS" in tables:
        for start, end, thread, text, text_id in connection.execute(
            "SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS"
        ):
            name = text or names.get(text_id, "")
            if thread is None or not name.startswith(range_prefix):
                continue
            process = thread & ~((1 << GLOBAL_THREAD_ID_BITS) - 1)
            if process not in all_processes:
                continue
            if start is None or end is None or end < start:
                incomplete_annotations[process] += 1
                continue
            add_duration(annotations[process], name, (end - start) / 1000)
    result = {
        "sqlite": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        },
        "analysis_script_sha256": file_sha256(Path(__file__)),
        "interval_implementation_sha256": file_sha256(Path(__file__).with_name("summarize_profile.py")),
        "scope": (
            "Per-device interval union from first through last recorded kernel or memory operation. CUDA API and NVTX "
            "durations are host elapsed intervals and may contain waiting. "
            "Summed kernel/API durations are not wall shares. "
            "No cross-host clock alignment, SM occupancy or useful-NCCL-transfer claim. "
            "Complete NVTX ranges are counted; capture edges can omit partial ranges. "
            "Profiler control APIs are reported separately and excluded from ordinary API interval unions."
        ),
        "memory_tables": memory_tables,
        "cuda_api_tables": api_tables,
        "nvtx_range_prefix": range_prefix,
        "devices": {
            str(device): {
                "gpu": gpu_properties[device],
                "event_counts": dict(counts[device]),
                "gpu_timeline": interval_times(intervals),
                "largest_gaps_without_recorded_gpu_work": longest_gaps(intervals),
                "kernels_by_summed_duration": ranked(kernels[device]),
                "processes": [
                    {
                        "global_pid": process,
                        **processes[process],
                        "cuda_api_by_summed_host_elapsed": ranked(api[process]),
                        "profiler_control_api_by_summed_host_elapsed": ranked(profiler_control_api[process]),
                        "cuda_api_interval_union": api_union(api_intervals[process]),
                        "nvtx_by_summed_host_elapsed": ranked(annotations[process]),
                        "complete_matching_nvtx_ranges": sum(row["count"] for row in annotations[process].values()),
                        "incomplete_matching_nvtx_ranges": incomplete_annotations[process],
                    }
                    for process in sorted(device_processes[device])
                ],
            }
            for device, intervals in sorted(devices.items())
        },
    }
    connection.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--range-prefix", default="execute_context_")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.sqlite, args.range_prefix)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("NSYS_SQLITE_SUMMARIZED", len(result["devices"]), args.output, flush=True)


if __name__ == "__main__":
    main()
