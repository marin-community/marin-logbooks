# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Require actual native worker CPU and memory binding before measurement."""

import hashlib
import json
import os
import urllib.request
from pathlib import Path


def main() -> None:
    profile = json.loads(Path(os.environ["KNOTMETER_PROFILE"]).read_text())
    output = Path(os.environ["KNOTMETER_CHECK_OUTPUT"])
    expected_hash = hashlib.sha256(Path(__file__).with_name("hero_numa_metadata.py").read_bytes()).hexdigest()
    body = {"method": "inspect_hero_numa", "timeout": 120}
    base = os.environ["KNOTMETER_SERVER_URL"].removesuffix("/v1")
    request = urllib.request.Request(
        f"{base}/collective_rpc", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        inspection = json.load(response)
    reports = inspection["results"]
    passed = bool(reports) and all(
        report["status"] == "passed"
        and report["extension_sha256"] == expected_hash
        and report["dp_size"] == profile["gpus"]["count"]
        and report["tp_size"] == 1
        for report in reports
    )
    report = {
        "status": "passed" if passed else "failed",
        "runtime_source": os.environ["KNOTMETER_RUNTIME_SOURCE"],
        "expected_extension_sha256": expected_hash,
        "inspection": inspection,
        "coverage": (
            "Pinned native DPLB utility broadcasts to every engine and propagates worker failures, "
            "but returns only the first engine result. Every rank writes actual CPU and memory-policy "
            "metadata under numa-affinity in its host archive. Retain and inspect all rank records."
        ),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise SystemExit("The actual native worker NUMA binding differs from the requested policy")


if __name__ == "__main__":
    main()
