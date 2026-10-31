# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare ordinal-matched collective kernel starts within each host."""

import argparse
import hashlib
import json
import sqlite3
import statistics
from pathlib import Path


def distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "sum": sum(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def analyze(path: Path) -> dict:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    names = dict(db.execute("SELECT id,value FROM StringIds"))
    kernels = {}
    for start, end, device, name in db.execute(
        "SELECT start,end,deviceId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
    ):
        if "nccl" in names[name].lower():
            kernels.setdefault(device, []).append((start, end, names[name]))
    assert len(kernels) == 4 and len(set(map(len, kernels.values()))) == 1
    devices = sorted(kernels)
    count = len(kernels[devices[0]])
    sequence = [row[2] for row in kernels[devices[0]]]
    assert all([row[2] for row in kernels[device]] == sequence for device in devices)
    assert count % 4 == 0
    assert all(("AllGather" in name) if i % 4 < 3 else ("Reduce_Sum" in name) for i, name in enumerate(sequence))
    positions = {}
    for i in range(count):
        rows = {device: kernels[device][i] for device in devices}
        latest = max(row[0] for row in rows.values())
        name = sequence[i]
        key = f"allgather_slot_{i % 4 + 1}" if "AllGather" in name else "reduce_slot_4"
        result = positions.setdefault(
            key,
            {
                "name": name,
                "count": 0,
                "start_skew_ms": [],
                "last_start_to_first_end_ms": [],
                "elapsed_before_latest_local_start_ms": {device: 0 for device in devices},
                "duration_ms": {device: 0 for device in devices},
            },
        )
        result["count"] += 1
        result["start_skew_ms"].append((latest - min(row[0] for row in rows.values())) / 1e6)
        result["last_start_to_first_end_ms"].append((min(row[1] for row in rows.values()) - latest) / 1e6)
        for device, (start, end, _) in rows.items():
            result["elapsed_before_latest_local_start_ms"][device] += max(0, min(end, latest) - start) / 1e6
            result["duration_ms"][device] += (end - start) / 1e6
    for result in positions.values():
        for key in ("start_skew_ms", "last_start_to_first_end_ms"):
            result[key] = distribution(result[key])
    db.close()
    return {
        "scope": (
            "Within one host only. The same ordered kernel-name sequence is verified on all four local GPUs. "
            "Ordinal matching is inferred from equal sequences, not an NCCL operation ID. "
            "Kernel residency before latest local start can include partial transfer; no pure-wait or bandwidth claim. "
            "Cross-host clocks are not aligned. Negative tail means kernels need not overlap on all local ranks."
        ),
        "sqlite_sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
        "analysis_sha256": hashlib.file_digest(Path(__file__).open("rb"), "sha256").hexdigest(),
        "collectives_per_device": count,
        "positions": positions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.sqlite)))


if __name__ == "__main__":
    main()
