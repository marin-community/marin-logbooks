# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract per-host identities from retained AgentX native outputs."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(records: Path) -> dict:
    points = []
    for concurrency in (8, 16, 32):
        directory = records / f"hero-c{concurrency}-native"
        hosts = []
        for status_path in sorted(directory.glob("host-*-native-status.json")):
            status = json.loads(status_path.read_text())
            native = status["attempts"][0]
            index = int(status["task_id"].rsplit("/", 1)[1])
            point = directory / f"host-{index}-{native['attempt_uid']}" / "point"
            attempt = json.loads((point / "attempt.json").read_text())
            logs = (point / "server-logs" / "stdout.log").read_text().splitlines()
            verified = [
                json.loads(line.split("MARIN_VLLM_WHEEL_VERIFIED=", 1)[1])
                for line in logs
                if "MARIN_VLLM_WHEEL_VERIFIED=" in line
            ]
            hosts.append(
                {
                    "task_index": attempt["task_index"],
                    "task_id": attempt["iris_task_id"],
                    "attempt_uid": attempt["iris_attempt_uid"],
                    "source_bundle": attempt["iris_bundle_id"],
                    "node": attempt["hostname"],
                    "gpu_uuids": attempt["gpu_uuids"],
                    "source": attempt["source"],
                    "native_archive": native["output_archive"],
                    "constraints_sha256": digest(point / "runtime-constraints.txt"),
                    "wheel_verified": verified,
                    "effective_rank_config": [line for line in logs if "GrugMoE effective config:" in line],
                    "graph_guard_observed": any("overriding cudagraph_mode to NONE" in line for line in logs),
                    "eager_disabled_observed": any("enforce_eager=False" in line for line in logs),
                    "server_command": attempt["server_command"],
                    "server_environment": attempt["server_environment"],
                    "collectors": attempt.get("collectors"),
                }
            )
        points.append({"concurrency": concurrency, "hosts": hosts})

    startup = records / "hero-startup-3a-native"
    startup_packages = []
    for path in sorted(startup.glob("*/point/runtime-packages.json")):
        packages = json.loads(path.read_text())
        startup_packages.append(
            {
                "host": path.parents[1].name,
                "inventory_sha256": digest(path),
                "inventory": packages,
            }
        )
    return {
        "points": points,
        "startup_package_inventories": startup_packages,
        "limits": [
            "Startup package inventories belong to the startup qualification, not the later C8/C16/C32 processes.",
            "The measured hosts retain commands, exact constraints, verified wheel URL/hash and extension paths, "
            "but no complete per-host package freeze or module checksums.",
            "Capture complete installed-package, Python-source and extension identities during baseline reproduction "
            "before changing the runtime.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.records)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for point in result["points"]:
        hosts = point["hosts"]
        print(
            f"C{point['concurrency']}: {len(hosts)} hosts, "
            f"{sum(len(host['gpu_uuids']) for host in hosts)} GPUs, "
            f"{sum(bool(host['wheel_verified']) for host in hosts)} verified wheels"
        )
    print(f"{len(result['startup_package_inventories'])} startup package inventories")


if __name__ == "__main__":
    main()
