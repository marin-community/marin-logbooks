# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare retained reference observations without allocating another model."""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
from marin.testing.inference.backend_parity import TokenDistribution, distribution_difference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--fp8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first, second = (json.loads(path.read_text()) for path in (args.bf16, args.fp8))
    assert first["model"] == second["model"]
    assert first["manifest"] == second["manifest"]
    assert first["policy"]["reference"] == second["policy"]["reference"]
    baseline = {(row["case"], row["position"]): row for row in first["checks"]["reference"]["observations"]}
    candidate = {(row["case"], row["position"]): row for row in second["checks"]["reference"]["observations"]}
    assert baseline.keys() == candidate.keys()
    rows = []
    for key, bf16 in baseline.items():
        fp8 = candidate[key]
        distributions = [TokenDistribution(**row["actual"]) for row in (bf16, fp8)]
        rows.append(
            {
                "case": key[0],
                "position": key[1],
                "bf16_reference_error": bf16["max_probability_error"],
                "fp8_reference_error": fp8["max_probability_error"],
                "fp8_vs_bf16": dataclasses.asdict(distribution_difference(*distributions)),
            }
        )
    groups = {}
    for group in ("short", "long"):
        selected = [row for row in rows if row["case"].startswith("long") == (group == "long")]
        assert len(selected) == first["checks"]["reference"]["groups"][group]["positions"]
        errors = np.asarray([row["fp8_vs_bf16"]["max_probability_error"] for row in selected])
        groups[group] = {
            "positions": len(selected),
            "pair_maximum": float(errors.max()),
            "pair_p99": float(np.quantile(errors, 0.99)),
            "pair_p90": float(np.quantile(errors, 0.9)),
            "different_greedy_fraction": (
                sum(not row["fp8_vs_bf16"]["same_greedy_token"] for row in selected) / len(selected)
            ),
            "worst_fp8_reference": sorted(selected, key=lambda row: row["fp8_reference_error"], reverse=True)[:10],
        }
    report = {
        "scope": (
            "Post-hoc numerical comparison of the two serving configurations; "
            "per-position top-k probability-error upper bounds, not KL or a new acceptance policy"
        ),
        "inputs": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (args.bf16, args.fp8)
        ],
        "runtime_sources": {"bf16": first["runtime_source"], "fp8": second["runtime_source"]},
        "groups": groups,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {group: {k: v for k, v in row.items() if k != "worst_fp8_reference"} for group, row in groups.items()}
        )
    )


if __name__ == "__main__":
    main()
