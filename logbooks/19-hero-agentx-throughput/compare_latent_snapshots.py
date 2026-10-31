# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare observed latent projections and a bounded independent FP32 check."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from compare_moe_snapshots import compare_tensor


def normalized(projected: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    value = projected.float()
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + epsilon)
    return value.to(projected.dtype) * weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {"reference": args.reference, "candidate": args.candidate}
    records = {
        label: [
            torch.load(root / f"latent-projection-{index}.pt", weights_only=True, map_location="cpu")
            for index in range(2)
        ]
        for label, root in inputs.items()
    }
    weights = {label: records[label][0]["weight"] for label in inputs}
    assert all(torch.isfinite(weight).all() for weight in weights.values())
    rows = []
    for index in range(2):
        reference, candidate = (records[label][index] for label in inputs)
        post = {
            label: torch.load(
                root / f"moe-snapshot-dp0-model.layers.0.mlp.experts-{index}.pt",
                weights_only=True,
                map_location="cpu",
            )["hidden"]
            for label, root in inputs.items()
        }
        changed = ((reference["projected"] != candidate["projected"]).any(dim=1)).nonzero().flatten().tolist()
        changed += ((post["reference"] != post["candidate"]).any(dim=1)).nonzero().flatten().tolist()
        selected = sorted(set([*range(8), *changed[:24]]))
        checks = {}
        for label in inputs:
            record = records[label][index]
            actual = record["projected"][selected]
            independent = record["input"][selected].float() @ weights[label].float().T
            checks[label] = {
                "projection_vs_fp32_rounded": compare_tensor(independent.to(actual.dtype), actual),
                "normalization_vs_cpu_reference": compare_tensor(
                    normalized(actual, record["norm_weight"], record["variance_epsilon"]), post[label][selected]
                ),
            }
        rows.append(
            {
                "prefill_index": index,
                "input": compare_tensor(reference["input"], candidate["input"]),
                "projected": compare_tensor(reference["projected"], candidate["projected"]),
                "norm_weight": compare_tensor(reference["norm_weight"], candidate["norm_weight"]),
                "normalized": compare_tensor(post["reference"], post["candidate"]),
                "sample_rows": selected,
                "independent_checks": checks,
            }
        )
    result = {
        "status": "completed",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "projection_weights_equal": torch.equal(weights["reference"], weights["candidate"]),
        "rows": rows,
        "scope": (
            "Full observed inputs and projection outputs; independent FP32 product on eight initial rows and "
            "up to24 changed-row entries. CPU normalization preserves BF16 cast before weight multiplication. "
            "Reports differences without introducing or changing a tolerance. No speed claim."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "projection_weights_equal": result["projection_weights_equal"]}))


if __name__ == "__main__":
    main()
