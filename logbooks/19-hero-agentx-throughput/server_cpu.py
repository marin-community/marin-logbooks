# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read CPU placement and process counters in one owned serving task.

This collector makes no CUDA calls and changes no process affinity. Keep it
separate from GPU profiling. Process lifetimes distinguish reused PIDs.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def snapshot() -> dict:
    processes = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            status = dict(line.split(":", 1) for line in (directory / "status").read_text().splitlines() if ":" in line)
            processes.append(
                {
                    "pid": int(directory.name),
                    "ppid": int(fields[1]),
                    "start_ticks": int(fields[19]),
                    "name": status["Name"].strip(),
                    "cpu_seconds": (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"),
                    "last_cpu": int(fields[36]),
                    "threads": int(status["Threads"]),
                    "rss_kib": status.get("VmRSS", "").strip(),
                    "cpus_allowed": status["Cpus_allowed_list"].strip(),
                    "mems_allowed": status["Mems_allowed_list"].strip(),
                }
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    cgroup = Path("/sys/fs/cgroup")
    return {
        "unix": time.time(),
        "processes": processes,
        "cgroup": {
            name: (cgroup / name).read_text().strip()
            for name in ("cpu.max", "cpu.stat", "cpu.pressure", "cpuset.cpus.effective", "cpuset.mems.effective")
            if (cgroup / name).exists()
        },
        "host_cpu_stat": Path("/proc/stat").read_text(),
        "host_load": Path("/proc/loadavg").read_text().strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=13)
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args()
    topology = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, check=False)
    with args.output.open("x", buffering=1) as stream:
        stream.write(
            json.dumps(
                {
                    "gpu_cpu_topology": topology.stdout,
                    "topology_returncode": topology.returncode,
                    "numa_cpu_lists": {
                        str(path): path.read_text().strip()
                        for path in Path("/sys/devices/system/node").glob("node*/cpulist")
                    },
                }
            )
            + "\n"
        )
        for index in range(args.samples):
            stream.write(json.dumps(snapshot()) + "\n")
            if args.attempt is not None and args.attempt.exists():
                if json.loads(args.attempt.read_text()).get("status") in ("succeeded", "failed"):
                    break
            if index + 1 < args.samples:
                time.sleep(5)


if __name__ == "__main__":
    main()
