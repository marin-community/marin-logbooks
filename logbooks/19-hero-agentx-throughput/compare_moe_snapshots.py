# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the retained CPU tensors at tiny-Hero routed-expert boundaries."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def compare_tensor(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    assert reference.shape == candidate.shape and reference.dtype == candidate.dtype
    assert torch.isfinite(reference).all() and torch.isfinite(candidate).all()
    delta = (reference.float() - candidate.float()).abs()
    different = (reference != candidate).nonzero()
    examples = []
    for index in different[:3].tolist():
        coordinate = tuple(index)
        examples.append(
            {"index": index, "reference": reference[coordinate].item(), "candidate": candidate[coordinate].item()}
        )
    return {
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "exact": torch.equal(reference, candidate),
        "different_elements": len(different),
        "elements": reference.numel(),
        "max_absolute_delta": delta.max().item(),
        "mean_absolute_delta": delta.mean().item(),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = sorted(path.name for path in args.reference.glob("moe-snapshot-*.pt"))
    assert len(names) == 8 and names == sorted(path.name for path in args.candidate.glob("moe-snapshot-*.pt"))
    rows = []
    for name in names:
        left, right = args.reference / name, args.candidate / name
        reference = torch.load(left, weights_only=True, map_location="cpu")
        candidate = torch.load(right, weights_only=True, map_location="cpu")
        assert reference["layer"] == candidate["layer"]
        rows.append(
            {
                "name": name,
                "reference_sha256": hashlib.sha256(left.read_bytes()).hexdigest(),
                "candidate_sha256": hashlib.sha256(right.read_bytes()).hexdigest(),
                "tensors": {
                    key: compare_tensor(reference[key], candidate[key]) for key in ["hidden", "router_logits", "output"]
                },
                "reference_strides": {
                    key: reference[key] for key in ["hidden_stride", "router_stride", "output_stride"]
                },
                "candidate_strides": {
                    key: candidate[key] for key in ["hidden_stride", "router_stride", "output_stride"]
                },
            }
        )
    result = {
        "status": "completed",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "rows": rows,
        "scope": "Two first-prompt prefills per tiny-Hero layer, DP0 only, replay disabled. No speed claim.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
