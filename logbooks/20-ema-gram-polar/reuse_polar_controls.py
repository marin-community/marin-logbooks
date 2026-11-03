# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reuse six covered o-product input routes, with exact report and HLO identity checks."""

import argparse
import hashlib
import json
from pathlib import Path

from compare_frontier_operands import compare


def reuse(current, prior, prior_bytes):
    expected = current["manifest"]["reports"]["frontier-operands"]
    assert hashlib.sha256(prior_bytes).hexdigest() == expected["sha256"]
    assert len(prior_bytes) == expected["bytes"]
    assert current["completed"] and prior["completed"]
    for name, graph in current["artifacts"].items():
        assert graph["sha256_verified"] and prior["artifacts"][name]["sha256_verified"]
        assert graph["input_sha256"] == prior["artifacts"][name]["input_sha256"]
    all_controls = compare(prior)
    keys = {f"o_proj/polar/{i}/B_X/{p}" for i in (2, 3, 4) for p in (0, 1)}
    result = {name: {key: rows[key] for key in sorted(keys)} for name, rows in all_controls["graphs"].items()}
    return dict(
        input_report=expected,
        comparator_source="compare_frontier_operands.py",
        graphs=result,
        limitations=[
            "Existing same-HLO o input traces reused; no new control extraction or execution.",
            "These are passing numerical controls in the prior full28leaf report, not newly measured outputs.",
            "Opaque boundaries and all prior mismatches retained; no runtime or causal inference.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.prior.read_bytes()
    result = reuse(json.loads(args.current.read_text()), json.loads(raw), raw)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
