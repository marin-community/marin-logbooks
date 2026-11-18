# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe an extracted native numactl on an owned ARM CPU coordinator."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from experiments.evaluation.knotmeter.artifacts import copy_file, write_record

PACKAGE = {
    "uri": (
        "s3://marin-us-east-08a/marin/users/romain/hero-agentx-perf-01a1846a/inputs/"
        "numactl-7141526de8f6d28fc3ebe69e35b6f6d0c9110734e40590676466c5772b076078/numactl_2.0.19-1_arm64.deb"
    ),
    "sha256": "7141526de8f6d28fc3ebe69e35b6f6d0c9110734e40590676466c5772b076078",
    "binary_sha256": "0648b4cfda388a18d8e100ca0b98295c3206a4482795d527b0ad5a028fb703b9",
    "provenance": (
        "marin-logbooks d6f5660563606d7877e7f40b8a70a59703a35f52, "
        "logbooks/19-hero-agentx-throughput/experiments/logical-kv-retention0-16-agentx-c128-r1.json"
    ),
}
ROOT = Path("/tmp/hero-bf16-numactl-7141526de8f6d28f")
STATUS_COMMAND = (
    "from pathlib import Path; import json; "
    "print(json.dumps([line for line in Path('/proc/self/status').read_text().splitlines() "
    "if line.startswith(('Cpus_allowed_list:', 'Mems_allowed_list:'))]))"
)


def prepare_numactl(directory: Path = ROOT) -> Path:
    """Extract the exact prior utility into a private task-local directory."""
    if platform.machine() not in {"aarch64", "arm64"}:
        raise ValueError("This retained package requires an ARM64 task")
    directory.mkdir(parents=True, exist_ok=True)
    package = directory / "numactl_2.0.19-1_arm64.deb"
    copy_file(PACKAGE["uri"], str(package))
    if hashlib.sha256(package.read_bytes()).hexdigest() != PACKAGE["sha256"]:
        raise ValueError("The numactl package differs from its retained digest")
    subprocess.run(["dpkg-deb", "--extract", str(package), str(directory / "root")], check=True)
    binary = directory / "root/usr/bin/numactl"
    if hashlib.sha256(binary.read_bytes()).hexdigest() != PACKAGE["binary_sha256"]:
        raise ValueError("The extracted numactl binary differs from its retained digest")
    return binary


def command_result(command: list[str]) -> dict:
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def probe(output_uri: str) -> None:
    binary = prepare_numactl()
    library = ctypes.CDLL("libnuma.so.1", use_errno=True)
    mode = ctypes.c_int()
    mempolicy_result = library.get_mempolicy(ctypes.byref(mode), None, ctypes.c_ulong(0), None, ctypes.c_ulong(0))
    mempolicy_errno = ctypes.get_errno()
    nodes = [
        {"node": int(path.name[4:]), "cpus": (path / "cpulist").read_text().strip()}
        for path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*"))
        if (path / "cpulist").read_text().strip()
    ]
    record = {
        "scope": "CPU-only dependency and native CPU/memory-policy probe. No CUDA context or model is created.",
        "hostname": platform.node(),
        "architecture": platform.machine(),
        "package": PACKAGE,
        "binary": str(binary),
        "uid": os.getuid(),
        "libnuma_available": library.numa_available(),
        "get_mempolicy_result": mempolicy_result,
        "get_mempolicy_errno": mempolicy_errno,
        "get_mempolicy_mode": mode.value,
        "process_status": [
            line
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith(("Cpus_allowed_list:", "Mems_allowed_list:"))
        ],
        "nodes_with_cpus": nodes,
        "hardware": command_result([str(binary), "--hardware"]),
        "version": command_result([str(binary), "--version"]),
        "bindings": [],
    }
    for node in nodes:
        node_id = str(node["node"])
        for memory in (True, False):
            flags = [f"--cpunodebind={node_id}"]
            if memory:
                flags += [f"--membind={node_id}"]
            record["bindings"].append(
                {"node": node["node"], "memory_binding": memory}
                | command_result([str(binary), *flags, sys.executable, "-c", STATUS_COMMAND])
            )
    write_record(output_uri, record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    probe(parser.parse_args().output)
