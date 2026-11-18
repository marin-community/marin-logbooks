# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read CPU and memory affinity from each actual native GPU worker."""

import ctypes
import hashlib
import json
import os
import platform
import threading
from collections import Counter
from pathlib import Path

NODE_MASK_BITS = 1024
WORD_BITS = ctypes.sizeof(ctypes.c_ulong) * 8
MPOL_BIND = 2


def memory_policy() -> dict:
    library = ctypes.CDLL("libnuma.so.1", use_errno=True)
    library.get_mempolicy.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
    ]
    library.get_mempolicy.restype = ctypes.c_int
    mode = ctypes.c_int()
    mask = (ctypes.c_ulong * (NODE_MASK_BITS // WORD_BITS))()
    ctypes.set_errno(0)
    result = library.get_mempolicy(ctypes.byref(mode), mask, NODE_MASK_BITS, None, 0)
    return {
        "returncode": result,
        "errno": ctypes.get_errno(),
        "mode": mode.value,
        "nodes": [node for node in range(NODE_MASK_BITS) if mask[node // WORD_BITS] & (1 << (node % WORD_BITS))],
        "scope": "get_mempolicy for the current thread; resident page placement is not inspected",
    }


def cpu_ids(value: str) -> set[int]:
    result = set()
    for entry in value.strip().split(","):
        first, separator, last = entry.partition("-")
        result.update(range(int(first), int(last) + 1) if separator else [int(first)])
    return result


class HeroNumaMetadata:
    def inspect_hero_numa(self) -> dict:
        parallel = self.vllm_config.parallel_config
        nodes = parallel.numa_bind_nodes
        if not parallel.numa_bind or nodes is None:
            raise ValueError("The native worker has no resolved NUMA binding")
        target = nodes[self.local_rank]
        expected_cpus = cpu_ids(Path(f"/sys/devices/system/node/node{target}/cpulist").read_text())
        actual_cpus = set(os.sched_getaffinity(0))
        policy = memory_policy()
        errors = []
        if not actual_cpus or not actual_cpus.issubset(expected_cpus):
            errors.append("The native worker execution thread is not confined to its GPU-local CPU node")
        if policy["returncode"] != 0 or policy["mode"] != MPOL_BIND or policy["nodes"] != [target]:
            errors.append("The native worker execution thread lacks the requested local memory policy")
        thread_masks = Counter()
        for thread in Path("/proc/self/task").iterdir():
            try:
                affinity = sorted(os.sched_getaffinity(int(thread.name)))
            except ProcessLookupError:
                continue
            thread_masks[",".join(str(cpu) for cpu in affinity)] += 1
        record = {
            "status": "failed" if errors else "passed",
            "errors": errors,
            "pid": os.getpid(),
            "native_thread_id": threading.get_native_id(),
            "hostname": platform.node(),
            "iris_node_name": os.environ.get("IRIS_NODE_NAME"),
            "iris_task_id": os.environ["IRIS_TASK_ID"],
            "iris_attempt_uid": os.environ["IRIS_ATTEMPT_UID"],
            "dp_rank": parallel.data_parallel_rank,
            "dp_size": parallel.data_parallel_size,
            "tp_size": parallel.tensor_parallel_size,
            "worker_local_gpu_index": self.local_rank,
            "worker_rank": self.rank,
            "assigned_physical_gpu_ids": parallel.assigned_physical_gpu_ids,
            "numa_bind": parallel.numa_bind,
            "numa_bind_nodes": nodes,
            "expected_numa_node": target,
            "expected_node_cpus": sorted(expected_cpus),
            "actual_thread_cpus": sorted(actual_cpus),
            "memory_policy": policy,
            "all_thread_cpu_masks": dict(thread_masks),
            "extension_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Read-only native worker metadata before warmup; no model tensor or cache state is changed",
        }
        path = Path(os.environ["IRIS_OUTPUT_DIR"]) / "point/numa-affinity" / f"dp-{parallel.data_parallel_rank}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n")
        if errors:
            raise ValueError("; ".join(errors))
        return record
