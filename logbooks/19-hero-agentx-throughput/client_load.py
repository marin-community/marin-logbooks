# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample one owned native benchmark client tree and its shared task cgroup.

Adapted from the benchmark owner's five-second AgentX sampler, SHA256
0999e1db9fb35c18f9108a12f4c946c33bba0bdeceef3892975c62a6f44c5c3e. The process selector targets native
bench/serve here. Process lifetime keys, CPU deltas and cgroup reads are unchanged.
Run one native benchmark client at a time in this owned task.
"""

import argparse
import json
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--duration", type=float, default=3600)
args = parser.parse_args()
output = args.output
clock_ticks = os.sysconf("SC_CLK_TCK")
page_size = os.sysconf("SC_PAGE_SIZE")
deadline = time.monotonic() + args.duration
previous_threads = {}
previous_time = time.monotonic()
seen_client = False
with output.open("x", buffering=1) as stream:
    while time.monotonic() < deadline:
        now = time.monotonic()
        candidates = []
        processes = {}
        for directory in Path("/proc").iterdir():
            if not directory.name.isdigit():
                continue
            try:
                fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
                command = (directory / "cmdline").read_bytes().split(b"\0")
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            pid = int(directory.name)
            processes[pid] = {"ppid": int(fields[1]), "rss": int(fields[21]) * page_size, "start": int(fields[19])}
            if (b"bench" in command and b"serve" in command) or (b"agentx" in command and b"--model" in command):
                candidates.append(pid)
        roots = [pid for pid in candidates if processes[pid]["ppid"] not in candidates]
        selected = set()
        if roots:
            selected.add(min(roots, key=lambda pid: processes[pid]["start"]))
            while True:
                descendants = {pid for pid, data in processes.items() if data["ppid"] in selected}
                if descendants <= selected:
                    break
                selected.update(descendants)
            seen_client = True
        thread_times = {}
        rss = 0
        for pid in selected:
            try:
                rss += processes[pid]["rss"]
                for thread in (Path("/proc") / str(pid) / "task").iterdir():
                    fields = (thread / "stat").read_text().rsplit(")", 1)[1].split()
                    thread_times[(pid, processes[pid]["start"], int(thread.name))] = (
                        int(fields[11]) + int(fields[12])
                    ) / clock_ticks
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
        interval = now - previous_time
        delta = {key: value - previous_threads[key] for key, value in thread_times.items() if key in previous_threads}
        stats = {}
        cgroup = Path("/sys/fs/cgroup")
        for name in ("cpu.stat", "cpu.max", "memory.current", "memory.max", "cpu.pressure"):
            path = cgroup / name
            if path.exists():
                stats[name] = path.read_text().strip()
        row = {
            "unix": time.time(),
            "interval_seconds": interval,
            "client_pids": sorted(selected),
            "client_cpu_cores": sum(delta.values()) / interval if delta else None,
            "client_busiest_thread_cores": max(delta.values()) / interval if delta else None,
            "client_threads": len(thread_times),
            "client_sum_rss_bytes": rss,
            "task_cgroup": stats,
            "network": Path("/proc/net/dev").read_text(),
        }
        stream.write(json.dumps(row) + "\n")
        previous_threads = thread_times
        previous_time = now
        if seen_client and not roots:
            break
        record = output.with_name("attempt.json")
        if record.exists() and json.loads(record.read_text()).get("status") in ("succeeded", "failed"):
            break
        time.sleep(5)
