# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare bounded declared product operands by group, endpoint kind and position."""

import argparse
import hashlib
import json
from pathlib import Path

from compare_outer_bindings import normalized, rooted_differences
from compare_start_producers import canonical_trace


def indexed(graph):
    result = {}
    groups = graph["trace"]["groups"]
    assert set(groups) == {"gate_proj", "down_proj", "up_proj", "o_proj"}
    for group, kinds in groups.items():
        assert set(kinds) == ({"product", "concat_declared_inputs"} if group in ("down_proj", "o_proj") else {"product"})
        for kind, endpoint in kinds.items():
            count = 2 if kind == "product" else 4
            assert sorted(x["operand_position"] for x in endpoint["operands"]) == list(range(count))
            for operand in endpoint["operands"]:
                key = f"{group}/{kind}/{operand['operand_position']}"
                assert key not in result
                result[key] = operand["trace"]
    assert len(result) == 16
    return result


def compare(report):
    assert report["completed"]
    graphs = report["artifacts"]
    reference = indexed(graphs["matrix_untouched"])
    result = {}
    for name, graph in graphs.items():
        assert graph["sha256_verified"]
        result[name] = {}
        for key, trace in indexed(graph).items():
            value, base = canonical_trace(trace), canonical_trace(reference[key])
            sig, _, _ = normalized(value, name)
            result[name][key] = dict(
                normalized_signature=sig,
                same_normalized_signature_as_untouched=sig == normalized(base, "matrix_untouched")[0],
                records=len(value["records"]),
                edges=len(value["edges"]),
                rooted_differences=rooted_differences(base, value, name),
                boundaries=[dict(**b, record=value["records"][b["node"]]) for b in value["boundaries"]],
            )
    assert all(v["same_normalized_signature_as_untouched"] for v in result["matrix_evaluate"].values())
    return dict(
        graphs=result,
        limitations=[
            "Endpoints matched by output dependency, group, kind and operand position; no text-order alignment.",
            "Traces stop at products, sources or opaque operations. Original Concat/Start/Done outputs stay opaque.",
            "Signatures include declarations, attributes, literals and ordered connections to boundaries.",
            "Recursive signatures do not distinguish sharing; raw edges retain caller contexts and repeated positions.",
            "Matching records do not establish values, precision, summation order, semantic identity or cause.",
            "Both numerical full28leaf substitution gates remain failed; no numerical intervention selected.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.sha256
    args.output.write_text(json.dumps(compare(json.loads(raw)), indent=2) + "\n")


if __name__ == "__main__":
    main()
