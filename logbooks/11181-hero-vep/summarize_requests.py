# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize private request evidence without exposing service identifiers."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def summarize(path: Path) -> dict:
    routes = Counter()
    statuses = Counter()
    completion_times = []
    usage = Counter()
    fingerprints = set()
    with path.open() as wire:
        for line in wire:
            record = json.loads(line)
            routes[record["route"]] += 1
            statuses[str(record.get("status", "transport failure"))] += 1
            if record["route"] != "v1/completions" or record.get("status") != 200:
                continue
            response = json.loads(record["response_text"])
            fingerprints.add(response["system_fingerprint"])
            completion_times.append(record["elapsed"])
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[field] += response["usage"][field]
    assert completion_times
    ordered = sorted(completion_times)
    return {
        "requests": sum(routes.values()),
        "routes": routes,
        "http_statuses": statuses,
        "completion_requests": len(ordered),
        "completion_latency_seconds": {
            "mean": math.fsum(ordered) / len(ordered),
            "p50": ordered[(len(ordered) - 1) // 2],
            "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "maximum": ordered[-1],
        },
        "usage": usage,
        "serving_fingerprint_count": len(fingerprints),
        "completion_usage_note": "Internal accounting only: max_tokens=0 and exact echoed input tokens were scored.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wire", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.wire)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
