# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile new product input roots against all saved bounded compiler reports."""

import argparse
import hashlib
import json
from pathlib import Path

from compare_frontier_operands import compare, indexed
from reconcile_compiler_coverage import compact, reconcile


def summarize(manifest, report):
    loaded = {}
    for label, entry in manifest["inputs"].items():
        raw = Path(entry["path"]).read_bytes()
        assert len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"]
        loaded[label] = json.loads(raw)
    inventory = loaded.pop("inventory")
    assert report["completed"]
    loaded["remaining-products"] = report
    graphs = {}
    for name in manifest["graphs"]:
        sha = inventory["manifest"]["artifacts"][name]["sha256"]
        assert inventory["artifacts"][name]["sha256_verified"]
        saved = {}
        for label, prior in loaded.items():
            if name not in prior["artifacts"]:
                assert label == "capture-initializer-concat" and name in ("matrix_untouched", "matrix_evaluate")
                continue
            artifact = prior["artifacts"][name]
            assert artifact["sha256_verified"] and artifact["input_sha256"] == sha
            saved[label] = artifact["trace"]
        graphs[name] = compact(reconcile(saved, inventory["artifacts"][name]))
    return dict(
        graphs=graphs,
        limitations=[
            "Exact-context complete input-root coverage; record presence is separate.",
            "Ordered repeated positions, projections and saved output dependencies retained.",
            "All original opaque outputs remain opaque even with covered inputs; no runtime graph or identity.",
            "Both full28leaf numerical substitution gates remain failed; no precision/order/values/cause claim.",
        ],
    )


def source_routes(report, summary):
    comparison = compare(report)
    graphs = {}
    for name, artifact in report["artifacts"].items():
        value = indexed(artifact)["o_proj/dependency/nonpolar_gram_direction/1"]
        rows = [value["records"][b["node"]] for b in value["boundaries"]]
        parameters = [row for row in rows if row["opcode"] == "parameter"]
        assert len(parameters) == 1
        graphs[name] = dict(
            right_state_parameter=parameters[0],
            right_route_boundaries=rows,
            new_product_gaps=[gap for gap in summary["graphs"][name]["gaps"] if gap["category"] == "product"],
            matching_routes=sum(
                row["same_normalized_signature_as_untouched"] for row in comparison["graphs"][name].values()
            ),
        )
    return dict(
        graphs=graphs,
        limitations=(
            "Declared state/source roles only; product traversal stops before operands; "
            "all original opaque outputs remain opaque. Matching bounded routes not runtime equality; "
            "both full28 numerical gates failed."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.sha256
    result = summarize(json.loads(args.manifest.read_text()), json.loads(raw))
    result["report_sha256"] = args.sha256
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    args.sources.write_text(json.dumps(source_routes(json.loads(raw), result), indent=2) + "\n")


if __name__ == "__main__":
    main()
